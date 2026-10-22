from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pagerank_lab import Graph
from pagerank_lab.cli import main
from pagerank_lab.generators import block_graph, skewed_graph, uniform_graph
from pagerank_lab.io import load_graph, load_vector, save_graph, save_result


@pytest.mark.parametrize("extension", ["npz", "mtx", "csv", "txt"])
def test_graph_roundtrip_keeps_weighted_edges_and_isolates(tmp_path, extension):
    graph = Graph.from_edges([0, 0, 1, 2], [1, 1, 2, 0], weights=[1, 2, 4, 5], n=6)
    path = tmp_path / f"graph.{extension}"
    save_graph(graph, path)
    restored = load_graph(path)
    np.testing.assert_array_equal(restored.adjacency.toarray(), graph.adjacency.toarray())
    assert restored.n == 6
    assert np.count_nonzero(restored.dangling) == 3


def test_csv_rejects_fractional_ids_and_invalid_weights(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("source,target,weight\n0.5,1,1\n")
    with pytest.raises(ValueError, match="line 2"):
        load_graph(path)
    for weight in ("NaN", "inf", "-1"):
        path.write_text(f"0,1,{weight}\n")
        with pytest.raises(ValueError, match="finite and nonnegative"):
            load_graph(path)


def test_explicit_nodes_cannot_disagree_with_metadata(tmp_path):
    path = tmp_path / "graph.csv"
    path.write_text("# n=5\n0,1\n")
    with pytest.raises(ValueError, match="declares"):
        load_graph(path, n=2)


def test_empty_edgelist_preserves_explicit_n(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# n=4\n")
    graph = load_graph(path)
    assert graph.n == 4 and graph.m == 0
    assert graph.dangling.all()


def test_numpy_vector_refuses_pickle(tmp_path):
    path = tmp_path / "object.npy"
    np.save(path, np.array([{"not": "numeric"}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_vector(path)


def test_numpy_vector_refuses_complex_values(tmp_path):
    path = tmp_path / "complex.npy"
    np.save(path, np.array([0.5 + 1j, 0.5]))
    with pytest.raises(ValueError, match="real"):
        load_vector(path)


@pytest.mark.parametrize("extension", ["json", "npz", "csv"])
def test_result_exports_values_and_metadata(tmp_path, extension):
    result = SimpleNamespace(scores=np.array([0.2, 0.3, 0.5]), iterations=4,
                             converged=True, residual_l1=1e-12,
                             error_bound_l1=7e-12, elapsed_seconds=0.01, history=[])
    path = tmp_path / f"scores.{extension}"
    save_result(result, path)
    if extension == "csv":
        values = np.loadtxt(path, delimiter=",", skiprows=1)
        np.testing.assert_array_equal(values[:, 0], [0, 1, 2])
        np.testing.assert_allclose(values[:, 1], result.scores)
        metadata = json.loads(path.with_suffix(".meta.json").read_text())
    elif extension == "json":
        np.testing.assert_allclose(load_vector(path, n=3), result.scores)
        metadata = json.loads(path.read_text())
    else:
        np.testing.assert_allclose(load_vector(path, n=3), result.scores)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
    assert metadata["converged"] is True
    assert metadata["iterations"] == 4


@pytest.mark.parametrize("factory", [uniform_graph, skewed_graph, block_graph])
def test_generators_are_reproducible_and_keep_reserved_dangling_nodes(factory):
    graph = factory(20, 100, seed=42, dangling_fraction=0.2, weighted=True)
    same = factory(20, 100, seed=42, dangling_fraction=0.2, weighted=True)
    assert (graph.adjacency != same.adjacency).nnz == 0
    assert graph.dangling[16:].all()
    assert graph.n == 20 and graph.m <= 100


def test_cli_generates_inspects_and_solves(tmp_path, capsys):
    path = tmp_path / "graph.npz"
    result = tmp_path / "scores.json"
    assert main(["generate", str(path), "--nodes", "16", "--edges", "80", "--seed", "5"]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["nodes"] == 16
    assert main(["inspect", str(path), "--components"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["strong_components"] >= 1
    assert main(["run", str(path), "--method", "scipy", "--output", str(result), "--top", "3"]) == 0
    solved = json.loads(capsys.readouterr().out)
    assert solved["converged"] is True
    assert len(solved["top"]) == 3
    np.testing.assert_allclose(load_vector(result).sum(), 1, atol=1e-12)


def test_cli_input_error_is_nonzero(tmp_path, capsys):
    assert main(["run", str(tmp_path / "missing.npz")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_nonconvergence_saves_diagnostic_result(tmp_path, capsys):
    graph = Graph.from_edges([0, 1], [1, 2], n=3)
    path = tmp_path / "graph.npz"
    output = tmp_path / "scores.json"
    save_graph(graph, path)
    assert main(["run", str(path), "--method", "scipy", "--max-iter", "1", "--output", str(output)]) == 2
    assert json.loads(capsys.readouterr().out)["converged"] is False
    assert json.loads(output.read_text())["converged"] is False

