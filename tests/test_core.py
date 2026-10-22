import numpy as np
import pytest
from scipy import sparse

from pagerank_lab import (Graph, pagerank, ConvergenceError, build_info,
                         personalized_batch, alpha_sweep)


def fixture_graph():
    return Graph.from_edges([0, 0, 1, 2, 2, 3], [1, 2, 2, 0, 2, 1], [2, 1, 3, 1, 1, 1], n=6)


def dense_solution(g, alpha, p, q):
    t = g.transition.toarray().T + np.outer(q, g.dangling)
    return np.linalg.solve(np.eye(g.n)-alpha*t, (1-alpha)*p)


@pytest.mark.parametrize("method", ["cython", "scipy"])
@pytest.mark.parametrize("alpha", [0., .2, .85, .99])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_weighted_disconnected_against_dense(method, alpha, index_dtype):
    g = Graph.from_scipy(fixture_graph().adjacency, index_dtype=index_dtype)
    p = np.array([0., 0., 0., .2, .3, .5])
    q = np.array([.5, .1, .1, .1, .1, .1])
    r = pagerank(g, alpha=alpha, personalization=p, dangling=q, method=method, max_iter=10000, tol=1e-11, history=True)
    exact = dense_solution(g, alpha, p, q)
    assert np.linalg.norm(r.scores-exact, 1) <= 1e-11
    assert r.error_bound_l1 <= r.tol
    assert abs(r.scores.sum()-1) < 1e-11
    assert len(r.history) == r.iterations


@pytest.mark.parametrize("method", ["cython", "scipy"])
def test_all_dangling_closed_form(method):
    g = Graph.from_edges([], [], n=4)
    p = np.array([.1, .2, .3, .4])
    q = p[::-1]
    r = pagerank(g, alpha=.7, personalization=p, dangling=q, method=method)
    np.testing.assert_allclose(r.scores, .3*p+.7*q, atol=1e-13)


def test_cycle_uniform_and_personalized_no_positive_teleport_requirement():
    g = Graph.from_edges([0, 1, 2], [1, 2, 0], n=4)
    p = np.array([1., 0., 0., 0.])
    r = pagerank(g, personalization=p, tol=1e-12)
    np.testing.assert_allclose(r.scores, dense_solution(g, .85, p, p), atol=1e-12)
    assert r.scores[3] == 0


def test_duplicates_zeros_and_small_weights():
    g = Graph.from_edges([0, 0, 0, 1], [1, 1, 2, 2], [1., 2., 0., 1e-320], n=3)
    assert g.m == 2
    assert g.adjacency[0, 1] == 3
    assert g.transition[1, 2] == 1
    assert np.all(np.isfinite(pagerank(g).scores))


@pytest.mark.parametrize("weights", [[-1.], [np.nan], [np.inf], [1j]])
def test_reject_bad_weights(weights):
    with pytest.raises(ValueError):
        Graph.from_edges([0], [1], weights, n=2)


def test_reject_negative_duplicate_before_canonicalization():
    a = sparse.coo_matrix(([-1., 3.], ([0, 0], [1, 1])), shape=(2, 2))
    with pytest.raises(ValueError):
        Graph.from_scipy(a)


@pytest.mark.parametrize("kwargs", [dict(alpha=1), dict(alpha=-1), dict(alpha=np.nan), dict(tol=0),
    dict(tol=np.inf), dict(threads=0), dict(max_iter=0), dict(dtype="float16"),
    dict(personalization=[0]*6), dict(personalization=[1, 2]), dict(x0=[-1]*6)])
def test_input_validation(kwargs):
    with pytest.raises(ValueError):
        pagerank(fixture_graph(), **kwargs)


def test_convergence_failure_returns_diagnostics():
    with pytest.raises(ConvergenceError) as exc:
        pagerank(fixture_graph(), max_iter=1, tol=1e-13)
    assert not exc.value.result.converged
    r = pagerank(fixture_graph(), max_iter=1, tol=1e-13, raise_on_failure=False)
    assert r.iterations == 1 and r.error_bound_l1 > r.tol


def test_float32_has_explicit_accuracy_limit():
    g = fixture_graph()
    r = pagerank(g, dtype="float32", tol=1e-6)
    assert r.scores.dtype == np.float32
    assert np.linalg.norm(r.scores-pagerank(g).scores, 1) < 1e-6
    r = pagerank(g, dtype="float32", tol=1e-13, raise_on_failure=False)
    assert not r.converged


def test_permutation_equivariance_and_warm_start():
    g = fixture_graph()
    order = np.array([5, 0, 3, 2, 1, 4])
    p = np.arange(1, 7)
    r = pagerank(g, personalization=p)
    reordered = pagerank(g.permute(order), personalization=p[order])
    np.testing.assert_allclose(reordered.scores, r.scores[order], atol=1e-12)
    warm = pagerank(g, personalization=p, x0=r.scores)
    assert warm.iterations <= 2


def test_row_scaling_invariance():
    g = fixture_graph()
    changed = Graph.from_scipy(sparse.diags([1e-5, 2., 1e5, 1., 1., 1.]) @ g.adjacency)
    np.testing.assert_allclose(pagerank(g).scores, pagerank(changed).scores, atol=1e-13)


def test_batch_and_sweep():
    g = fixture_graph()
    ps = np.eye(6)[:2]
    batch = personalized_batch(g, ps)
    np.testing.assert_allclose(batch[0].scores, pagerank(g, personalization=ps[0]).scores)
    results = alpha_sweep(g, [0, .5, .85])
    assert len(results) == 3
    np.testing.assert_allclose(results[-1].scores, pagerank(g).scores, atol=1e-10)


def test_networkx_correspondence():
    nx = pytest.importorskip("networkx")
    g = fixture_graph()
    ng = nx.from_scipy_sparse_array(g.adjacency, create_using=nx.DiGraph)
    expected = nx.pagerank(ng, tol=1e-14, max_iter=10000)
    np.testing.assert_allclose(pagerank(g, tol=1e-12).scores, list(expected.values()), atol=1e-12)
    assert Graph.from_networkx(ng).labels == tuple(range(6))


def test_openmp_matches_serial():
    if not build_info()["openmp"]:
        pytest.skip("OpenMP not compiled")
    rng = np.random.default_rng(1942)
    g = Graph.from_edges(rng.integers(0, 1000, 12000), rng.integers(0, 1000, 12000), n=1000)
    for dtype, tol in [("float64", 1e-11), ("float32", 1e-6)]:
        a = pagerank(g, dtype=dtype, tol=tol, threads=1)
        b = pagerank(g, dtype=dtype, tol=tol, threads=4)
        np.testing.assert_allclose(a.scores, b.scores, atol=tol)
        rows = pagerank(g, dtype=dtype, tol=tol, threads=4, partition="rows")
        np.testing.assert_allclose(a.scores, rows.scores, atol=tol)


def test_overflow_normalization_and_readonly_scores():
    r = pagerank(fixture_graph(), personalization=np.full(6, 1e308))
    assert abs(r.scores.sum()-1) < 1e-10
    assert not r.scores.flags.writeable


def test_alpha_zero_preserves_dtype_and_error_target():
    r = pagerank(fixture_graph(), alpha=0, dtype="float32", tol=1e-6)
    assert r.scores.dtype == np.float32
    assert r.error_bound_l1 <= r.tol
