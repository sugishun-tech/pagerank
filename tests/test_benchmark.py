"""Tests of fair comparison, independently checked output, and artifact schema."""
import json

import numpy as np
import pytest

from benchmarks.run import (
    Config, independent_check, make_adjacency, networkx_tolerance,
    run_suite, summarize_samples, write_report, python_power,
)


def test_networkx_tolerance_matches_absolute_bound():
    alpha, n, target = .85, 100000, 1e-10
    nx_tol = networkx_tolerance(target, alpha, n)
    assert alpha/(1-alpha) * n * nx_tol == pytest.approx(target)
    assert networkx_tolerance(target, 0, n) == target/n
    with pytest.raises(ValueError):
        networkx_tolerance(target, 1, n)


@pytest.mark.parametrize("family", ["uniform", "skewed", "block", "dangling", "cycle"])
def test_generator_reproducible_and_nonnegative(family):
    a, meta = make_adjacency(17, 8, family, 42, weighted=True)
    b, meta2 = make_adjacency(17, 8, family, 42, weighted=True)
    assert (a != b).nnz == 0
    assert meta == meta2
    assert (a.data > 0).all()
    if family == "dangling":
        assert np.asarray(a.sum(axis=1)).ravel()[::5].sum() == 0


def test_independent_residual_rejects_wrong_stationary_vector():
    from pagerank_lab import Graph
    graph = Graph.from_edges([0], [1], n=2)
    p = np.array([.5, .5])
    # Analytic solution for 0 -> 1 with uniform dangling redistribution.
    alpha = .85
    exact = np.array([1/(2+alpha), (1+alpha)/(2+alpha)])
    assert independent_check(graph, exact, alpha, p, p, 1e-12)["valid"]
    assert not independent_check(graph, p, alpha, p, p, 1e-12)["valid"]


def test_summary_preserves_failure():
    samples = [dict(wall_seconds=.1, valid=True, error_bound_l1=1e-12, mass_error=0,iterations=10),
               dict(wall_seconds=.2, valid=False, error_bound_l1=1e-3,mass_error=0,iterations=3)]
    result = summarize_samples(samples)
    assert result["median_seconds"] == pytest.approx(.15)
    assert not result["valid"]
    assert result["max_error_bound_l1"] == 1e-3


def test_plain_python_matches_analytic_solution():
    scores, details = python_power([[],[(0,1.0)]],[1],.85,[.5,.5],[.5,.5],1e-12,1000)
    np.testing.assert_allclose(scores,[1/2.85,1.85/2.85],atol=1e-12,rtol=0)
    assert details["iterations"]>0


def test_nonfinite_failure_still_serializes(tmp_path):
    from pagerank_lab import Graph
    graph = Graph.from_edges([0], [1], n=2)
    p = np.array([.5,.5])
    check = independent_check(graph,[1.7e308,1.7e308],.85,p,p,1e-12)
    assert not check["valid"]
    summary = summarize_samples([{**check,"wall_seconds":.01}])
    json.dumps(summary, allow_nan=False)


def test_small_comparison_and_report(tmp_path):
    pytest.importorskip("networkx")
    config = Config(sizes=(30,),families=("dangling",),methods=("scipy","networkx"),
                    alphas=(.85,),threads=(1,),repeats=2,warmups=1,degree=4,
                    personalization="random",dangling_distribution="random")
    report=run_suite(config)
    case=report["cases"][0]
    assert all(item["status"]=="ok" for item in case["methods"].values())
    assert len(case["execution_orders"])==2
    json_file,csv_file=write_report(report,tmp_path)
    assert json.loads(json_file.read_text())["schema_version"]==1
    assert "networkx" in csv_file.read_text()


def test_linear_methods_and_size_caps():
    config = Config(sizes=(12,),families=("dangling",),methods=("direct","gmres","python"),
                    repeats=1,warmups=0,degree=3,python_max_n=10)
    report = run_suite(config)
    methods = report["cases"][0]["methods"]
    assert methods["python"]["status"] == "skipped"
    for name in ("direct","gmres"):
        assert methods[name]["status"] == "ok"
        assert methods[name]["samples"][0]["iterations"] is None
    skipped = run_suite(Config(sizes=(12,),families=("dangling",),methods=("direct",),
                               repeats=1,warmups=0,degree=3,direct_max_n=10))
    assert skipped["cases"][0]["methods"]["direct"]["status"] == "skipped"


@pytest.mark.parametrize("config",[Config(sizes=(0,)),Config(alphas=(1.,)),Config(repeats=0),Config(tol=float("nan"))])
def test_config_rejects_invalid(config):
    with pytest.raises(ValueError):
        config.validate()
