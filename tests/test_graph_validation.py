"""Conversion-boundary regressions, including a former native crash case."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy import sparse

from pagerank_lab import Graph
from pagerank_lab.io import load_graph


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "lil", "dok", "dia", "bsr"])
@pytest.mark.parametrize("kind", ["matrix", "array"])
def test_supported_scipy_formats(format, kind):
    constructor = getattr(sparse, f"{format}_{kind}", None)
    if constructor is None:
        pytest.skip("sparse array format unavailable in this SciPy version")
    expected = np.array([[0., 2., 1.], [1., 0., 0.], [0., 0., 0.]])
    graph = Graph.from_scipy(constructor(expected))
    np.testing.assert_array_equal(graph.adjacency.toarray(), expected)
    np.testing.assert_allclose(graph.transition[0].toarray(), [[0., 2/3, 1/3]])
    assert graph.dangling.tolist() == [False, False, True]


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "lil", "dok", "dia", "bsr"])
def test_every_sparse_format_rejects_negative_weights(format):
    matrix = getattr(sparse, format + "_matrix")([[0., -1.], [0., 0.]])
    with pytest.raises(ValueError, match="nonnegative"):
        Graph.from_scipy(matrix)


@pytest.mark.parametrize("indptr", [[0, 2, 1], [0, -1, 1], [0, 1, 100]])
def test_bad_compressed_row_pointers_are_rejected_before_conversion(indptr):
    matrix = sparse.csc_matrix(([1.], [0], [0, 1, 1]), shape=(2, 2))
    matrix.indptr = np.array(indptr, dtype=np.int32)
    with pytest.raises(ValueError, match="indptr"):
        Graph.from_scipy(matrix)


def test_bad_coo_coordinate_is_rejected_before_conversion():
    matrix = sparse.coo_matrix(([1.], ([0], [1])), shape=(2, 2))
    matrix.row[0] = 100000
    with pytest.raises(ValueError, match="indices"):
        Graph.from_scipy(matrix)


def test_bad_lil_row_lengths_are_rejected_before_native_flattening():
    matrix = sparse.lil_matrix((2, 2))
    matrix.rows[0] = [0, 1]
    matrix.data[0] = [1.]
    with pytest.raises(ValueError, match="equal lengths"):
        Graph.from_scipy(matrix)


def test_malformed_csc_does_not_crash_subprocess(tmp_path):
    # Before source validation, CSC->CSR reached scipy's native csr_tocsc
    # and exited with SIGSEGV. Keep this boundary regression process-isolated.
    program = textwrap.dedent("""
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except ImportError:
            pass
        from scipy import sparse
        from pagerank_lab import Graph
        matrix = sparse.csc_matrix(([1.], [100000], [0, 1, 1]), shape=(2, 2))
        try:
            Graph.from_scipy(matrix)
        except ValueError:
            print("rejected")
        else:
            raise AssertionError("malformed CSC accepted")
    """)
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True,
                               text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "rejected"


def test_malformed_csc_npz_is_rejected_before_io_conversion(tmp_path):
    # Write the portable sparse NPZ schema directly, without converting input.
    path = tmp_path / "malformed.npz"
    np.savez(path, format=np.array(b"csc"), shape=np.array([2, 2]),
             data=np.array([1.]), indices=np.array([100000]), indptr=np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="indices"):
        load_graph(path)


def test_source_arrays_are_not_mutated():
    matrix = sparse.csc_matrix(([1., 2.], [1, 0], [0, 2, 2]), shape=(2, 2))
    before = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
    Graph.from_scipy(matrix)
    for original, current in zip(before, (matrix.data, matrix.indices, matrix.indptr)):
        np.testing.assert_array_equal(current, original)

