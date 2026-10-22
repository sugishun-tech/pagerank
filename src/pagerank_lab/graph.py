"""Validated, reusable sparse directed graphs (row i contains edges i -> j)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _validated_sparse_values(matrix):
    """Validate raw source storage before any compiled sparse conversion.

    SciPy constructors do not always perform a full index check. In
    particular, converting malformed CSC/COO input to CSR can reach native
    code before CSR.check_format is called. These checks do not mutate the
    caller's arrays or trust cached sorted/canonical-format flags.
    """
    n = matrix.shape[0]
    format = matrix.format

    def indices(values, limit, name):
        values = np.asarray(values)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if np.any(values < 0) or np.any(values >= limit):
            raise ValueError(f"{name} entries must lie in [0, {limit})")
        return values

    if format in {"csr", "csc", "bsr"}:
        data = np.asarray(matrix.data)
        major = minor = n
        if format == "bsr":
            if data.ndim != 3 or data.shape[1] < 1 or data.shape[2] < 1:
                raise ValueError("BSR data must contain positive-size 2D blocks")
            rows, cols = data.shape[1:]
            if n % rows or n % cols:
                raise ValueError("BSR shape must be divisible by its block dimensions")
            major, minor = n // rows, n // cols
        elif data.ndim != 1:
            raise ValueError("CSR/CSC data must be one-dimensional")
        cols = indices(matrix.indices, minor, "sparse indices")
        ptr = np.asarray(matrix.indptr)
        if ptr.ndim != 1 or ptr.dtype.kind not in "iu" or len(ptr) != major + 1:
            raise ValueError("invalid sparse indptr shape or dtype")
        if len(cols) != len(data):
            raise ValueError("sparse indices and data must have equal lengths")
        if (ptr[0] != 0 or np.any(ptr < 0) or np.any(ptr > len(cols))
                or np.any(ptr[1:] < ptr[:-1])):
            raise ValueError("sparse indptr must start at zero, be nondecreasing, and stay within storage")
        return data
    if format == "coo":
        data = np.asarray(matrix.data)
        rows = indices(matrix.row, n, "COO row indices")
        cols = indices(matrix.col, n, "COO column indices")
        if data.ndim != 1 or len(rows) != len(cols) or len(rows) != len(data):
            raise ValueError("COO coordinates and data must have matching one-dimensional shapes")
        return data
    if format == "lil":
        if np.shape(matrix.rows) != (n,) or np.shape(matrix.data) != (n,):
            raise ValueError("LIL storage must contain one entry per matrix row")
        chunks = []
        for row_indices, row_values in zip(matrix.rows, matrix.data):
            values = np.asarray(row_values)
            row_indices = np.asarray(row_indices)
            # An empty Python list acquires float64 dtype; no IDs exist to test.
            if row_indices.size:
                indices(row_indices, n, "LIL column indices")
            elif row_indices.ndim != 1:
                raise ValueError("LIL column indices must be one-dimensional")
            if values.ndim != 1 or len(row_indices) != len(values):
                raise ValueError("LIL row indices and values must have equal lengths")
            chunks.append(values)
        return np.concatenate(chunks) if chunks else np.empty(0)
    if format == "dok":
        values = []
        for coordinate, value in matrix.items():
            if not isinstance(coordinate, tuple) or len(coordinate) != 2:
                raise ValueError("DOK keys must be (row, column) pairs")
            for item in coordinate:
                if _integer(item, "DOK coordinate") >= n:
                    raise ValueError("DOK coordinate is outside the matrix shape")
            values.append(value)
        return np.asarray(values)
    if format == "dia":
        data, offsets = np.asarray(matrix.data), np.asarray(matrix.offsets)
        if (data.ndim != 2 or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
                or data.shape[0] != len(offsets) or len(np.unique(offsets)) != len(offsets)):
            raise ValueError("invalid DIA data or diagonal offsets")
        return data
    raise ValueError(f"unsupported sparse format: {format}")


@dataclass(frozen=True)
class PreparedGraph:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    dangling: np.ndarray


class Graph:
    """Nonnegative weighted graph, with duplicate edges summed and zeros removed.

    Public sparse arrays are read-only. Treat the entire graph as immutable;
    create a new Graph when changing edges. Never replace its sparse arrays or
    structural attributes: prepared native buffers assume the validated graph
    stays unchanged. Node order is always explicit.
    """

    __slots__ = ("n", "m", "adjacency", "transition", "incoming", "dangling",
                 "out_strength", "labels", "_prepared")

    def __init__(self, matrix, *, labels=None, index_dtype="auto"):
        if getattr(matrix, "ndim", 2) != 2:
            raise ValueError("adjacency must be a two-dimensional square matrix")
        if sparse.issparse(matrix):
            if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
                raise ValueError("adjacency must be square with at least one node")
            raw = _validated_sparse_values(matrix)
        else:
            raw = np.asarray(matrix)
        if np.iscomplexobj(raw):
            raise ValueError("complex edge weights are not supported")
        try:
            raw_values = np.asarray(raw, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("weights must be real numbers") from exc
        if not np.all(np.isfinite(raw_values)) or np.any(raw_values < 0):
            raise ValueError("weights must be finite and nonnegative")
        a = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
        if a.shape[0] != a.shape[1] or a.shape[0] < 1:
            raise ValueError("adjacency must be square with at least one node")
        a.check_format(full_check=True)
        a.sum_duplicates()
        a.eliminate_zeros()
        a.sort_indices()
        if not np.all(np.isfinite(a.data)):
            raise ValueError("summed edge weights overflowed float64")
        self.n, self.m = int(a.shape[0]), int(a.nnz)
        if labels is None:
            self.labels = None
        else:
            self.labels = tuple(labels)
            if len(self.labels) != self.n or len(set(self.labels)) != self.n:
                raise ValueError("labels must contain one distinct label per node")
        if index_dtype not in ("auto", "int32", "int64"):
            raise ValueError("index_dtype must be auto, int32, or int64")
        fits32 = max(self.n, self.m) <= np.iinfo(np.int32).max
        if index_dtype == "int32" and not fits32:
            raise ValueError("graph is too large for int32 indices")
        idx = np.int32 if (index_dtype == "int32" or index_dtype == "auto" and fits32) else np.int64
        with np.errstate(over="ignore"):
            self.out_strength = np.asarray(a.sum(axis=1)).ravel()
        if not np.all(np.isfinite(self.out_strength)):
            raise ValueError("outgoing weight sum overflowed; rescale weights in each row")
        self.dangling = self.out_strength == 0
        h = a.copy()
        # Row-wise division avoids overflow of 1/strength for subnormal weights.
        h.data /= np.repeat(np.where(self.dangling, 1., self.out_strength), np.diff(h.indptr))
        c = h.T.tocsr()
        for mat in (a, h, c):
            mat.indptr = np.ascontiguousarray(mat.indptr, dtype=idx)
            mat.indices = np.ascontiguousarray(mat.indices, dtype=idx)
            for buf in (mat.indptr, mat.indices, mat.data):
                buf.flags.writeable = False
        self.adjacency, self.transition, self.incoming = a, h, c
        self.out_strength.flags.writeable = False
        self.dangling.flags.writeable = False
        self._prepared = {}

    @classmethod
    def from_scipy(cls, matrix, **kwargs):
        return cls(matrix, **kwargs)

    @classmethod
    def from_edges(cls, sources, targets, weights=None, n=None, **kwargs):
        src, dst = np.asarray(sources), np.asarray(targets)
        if src.ndim != 1 or dst.ndim != 1 or src.size != dst.size:
            raise ValueError("sources and targets must be equally sized one-dimensional arrays")
        for arr in (src, dst):
            if arr.size and (not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0)):
                raise ValueError("node IDs must be nonnegative integers")
        if n is None:
            if not src.size:
                raise ValueError("supply n >= 1 for an edgeless graph")
            n = max(int(src.max()), int(dst.max())) + 1
        n = _integer(n, "n", 1)
        if src.size and (int(src.max()) >= n or int(dst.max()) >= n):
            raise ValueError("node ID is outside [0, n)")
        if weights is None:
            values = np.ones(src.size, dtype=np.float64)
        else:
            if np.iscomplexobj(weights):
                raise ValueError("complex edge weights are not supported")
            values = np.asarray(weights, dtype=np.float64)
            if values.ndim != 1 or values.size != src.size:
                raise ValueError("weights must have one entry per edge")
            if np.any(values < 0) or not np.all(np.isfinite(values)):
                raise ValueError("weights must be finite and nonnegative")
        a = sparse.coo_matrix((values, (src.astype(np.int64), dst.astype(np.int64))), shape=(n, n))
        return cls(a, **kwargs)

    @classmethod
    def from_networkx(cls, graph, nodelist=None, weight="weight", **kwargs):
        import networkx as nx
        labels = list(graph) if nodelist is None else list(nodelist)
        if len(labels) != len(graph) or set(labels) != set(graph):
            raise ValueError("nodelist must be a permutation of all nodes")
        for _, _, data in graph.edges(data=True):
            value = 1. if weight is None else data.get(weight, 1.)
            if not np.isfinite(value) or value < 0:
                raise ValueError("weights must be finite and nonnegative")
        a = nx.to_scipy_sparse_array(graph, nodelist=labels, weight=weight, dtype=np.float64, format="csr")
        return cls(a, labels=labels, **kwargs)

    def prepare(self, dtype="float64") -> PreparedGraph:
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        key = dtype.str
        if key not in self._prepared:
            data = np.ascontiguousarray(self.incoming.data, dtype=dtype)
            dangling = np.ascontiguousarray(self.dangling, dtype=np.uint8)
            data.flags.writeable = dangling.flags.writeable = False
            self._prepared[key] = PreparedGraph(self.incoming.indptr, self.incoming.indices, data, dangling)
        return self._prepared[key]

    def partition(self, threads, strategy="balanced"):
        """Contiguous destination-row partitions; balance nonzeros plus row cost."""
        threads = _integer(threads, "threads", 1)
        if strategy not in ("balanced", "rows"):
            raise ValueError("partition must be balanced or rows")
        key = ("partition", threads, strategy)
        if key not in self._prepared:
            if strategy == "rows":
                cuts = np.linspace(0, self.n, threads+1, dtype=np.int64)
            else:
                cost = self.incoming.indptr.astype(np.int64) + np.arange(self.n+1, dtype=np.int64)
                targets = np.linspace(0., float(cost[-1]), threads+1)
                cuts = np.searchsorted(cost, targets).astype(np.int64)
                cuts[0], cuts[-1] = 0, self.n
            cuts.flags.writeable = False
            self._prepared[key] = cuts
        return self._prepared[key]

    def reverse(self):
        """Reverse edges, then recompute outgoing normalization."""
        return Graph(self.adjacency.T, labels=self.labels)

    def permute(self, order):
        """Return an equivalent graph in the node order given by a permutation."""
        order = np.asarray(order)
        if order.shape != (self.n,) or not np.issubdtype(order.dtype, np.integer) or not np.array_equal(np.sort(order), np.arange(self.n)):
            raise ValueError("order must be a permutation of 0..n-1")
        labels = None if self.labels is None else [self.labels[i] for i in order]
        return Graph(self.adjacency[order][:, order], labels=labels)

    def statistics(self, *, components=False):
        """Graph/storage diagnostics; component analysis is opt-in."""
        indeg, outdeg = np.diff(self.incoming.indptr), np.diff(self.adjacency.indptr)
        result = {
            "n": self.n, "m": self.m, "dangling_nodes": int(self.dangling.sum()),
            "self_loops": int(np.count_nonzero(self.adjacency.diagonal())),
            "mean_degree": self.m / self.n,
            "max_in_degree": int(indeg.max()), "max_out_degree": int(outdeg.max()),
            "index_dtype": str(self.incoming.indices.dtype),
            "sparse_storage_bytes": sum(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes for x in (self.adjacency, self.transition, self.incoming)),
        }
        if components:
            from scipy.sparse.csgraph import connected_components
            result["strong_components"] = int(connected_components(self.adjacency, directed=True, connection="strong", return_labels=False))
            result["weak_components"] = int(connected_components(self.adjacency, directed=True, connection="weak", return_labels=False))
        return result

    def __repr__(self):
        return f"Graph(n={self.n}, m={self.m}, dangling={int(self.dangling.sum())})"
