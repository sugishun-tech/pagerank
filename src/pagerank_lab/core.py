"""PageRank solvers with explicit accuracy and reproducibility semantics."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .graph import Graph, _integer


def probability_vector(value, n, name="vector"):
    """Normalize finite nonnegative weights without overflowing their sum."""
    if value is None:
        return np.full(n, 1. / n, dtype=np.float64)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    x = np.array(value, dtype=np.float64, copy=True)
    if x.shape != (n,) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError(f"{name} must have shape ({n},) with finite nonnegative entries")
    scale = x.max()
    if scale <= 0:
        raise ValueError(f"{name} must have positive mass")
    x /= scale
    x /= x.sum()
    return x


@dataclass(frozen=True)
class PageRankResult:
    scores: np.ndarray
    iterations: int
    converged: bool
    residual_l1: float
    error_bound_l1: float
    delta_l1: float
    elapsed_seconds: float
    alpha: float
    tol: float
    method: str
    dtype: str
    threads: int
    history: np.ndarray | None = None
    partition: str = "balanced"

    def topk(self, k=10):
        """(node index, score), descending score; node index breaks exact ties."""
        k = _integer(k, "k", 0)
        order = np.lexsort((np.arange(self.scores.size), -self.scores))[:k]
        return [(int(i), float(self.scores[i])) for i in order]

    def as_dict(self, labels=None):
        if labels is None:
            labels = range(self.scores.size)
        labels = list(labels)
        if len(labels) != self.scores.size:
            raise ValueError("labels must match the number of scores")
        return dict(zip(labels, map(float, self.scores)))

    def metadata(self):
        return {key: value for key, value in vars(self).items() if key not in ("scores", "history")}


class ConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"PageRank did not meet L1 target {result.tol:g}: residual bound "
                         f"{result.error_bound_l1:.4g} after {result.iterations} iterations "
                         f"({result.dtype}); increase max_iter or use float64 / a looser tol")


def build_info():
    try:
        from . import _kernels
    except ImportError:
        return {"available": False, "openmp": False, "native": False}
    return {"available": True, **_kernels.build_info()}


def pagerank(graph: Graph, *, alpha=0.85, personalization=None, dangling=None,
             tol=1e-10, max_iter=1000, method="cython", dtype="float64", threads=1,
             x0=None, history=False, raise_on_failure=True, partition="balanced") -> PageRankResult:
    """Compute the unique PageRank for 0 <= alpha < 1.

    ``tol`` is an absolute L1 error target, independent of graph size. The
    posterior residual is evaluated in float64 against the float64 graph, even
    for float32 iteration. The bound is mathematical, not interval arithmetic.
    ``elapsed_seconds`` measures only the solver, excluding validation, graph
    preparation and the final residual check; benchmark full calls separately.
    ``raise_on_failure=False`` returns the last power iterate. Direct/GMRES
    failures still raise because those solvers do not expose a failed iterate.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a pagerank_lab.Graph")
    alpha, tol = float(alpha), float(tol)
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    max_iter = _integer(max_iter, "max_iter", 1)
    threads = _integer(threads, "threads", 1)
    if max_iter > np.iinfo(np.int32).max or threads > np.iinfo(np.int32).max:
        raise ValueError("max_iter and threads must fit a C int")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if method not in ("cython", "scipy", "direct", "gmres"):
        raise ValueError("method must be cython, scipy, direct, or gmres")
    if partition not in ("balanced", "rows"):
        raise ValueError("partition must be balanced or rows")
    if method != "cython" and threads != 1:
        raise ValueError("threads applies only to the Cython implementation")
    if method in ("direct", "gmres") and dtype != np.dtype("float64"):
        raise ValueError("linear solvers use float64")
    n = graph.n
    p = probability_vector(personalization, n, "personalization")
    q = p if dangling is None else probability_vector(dangling, n, "dangling")
    initial = probability_vector(x0, n, "x0")
    same_pq = dangling is None or np.array_equal(p, q)
    uniform = same_pq and bool(np.all(p == 1./n))
    record = np.empty(max_iter, dtype=np.float64) if history else np.empty(0, dtype=np.float64)
    threshold = tol * (1-alpha) / alpha if alpha else np.inf
    delta, iterations = 0., 0
    if alpha == 0:
        start = perf_counter()
        scores = p.astype(dtype)
        elapsed = perf_counter() - start
    elif method == "cython":
        try:
            from . import _kernels
        except ImportError as exc:
            raise ImportError("Cython extension is unavailable; install the package or use method='scipy'") from exc
        if threads > 1 and not _kernels.build_info()["openmp"]:
            raise ValueError("threads > 1 requires a build with PAGERANK_OPENMP=1")
        prepared = graph.prepare(dtype)
        cuts = graph.partition(threads, partition)
        scratch = np.empty(16*threads, dtype=np.float64)
        first = initial.astype(dtype)
        second = np.empty_like(first)
        pp, qq = p.astype(dtype), q.astype(dtype)
        start = perf_counter()
        iterations, delta = _kernels.power(prepared.indptr, prepared.indices, prepared.data,
            prepared.dangling, pp, qq, first, second, alpha, threshold, max_iter,
            threads, uniform, same_pq, record, cuts, scratch)
        elapsed = perf_counter() - start
        scores = second if iterations % 2 else first
    elif method == "scipy":
        h = graph.incoming.astype(dtype, copy=False)
        x = initial.astype(dtype)
        pp, qq = p.astype(dtype), q.astype(dtype)
        start = perf_counter()
        for iterations in range(1, max_iter + 1):
            dm = float(np.sum(x[graph.dangling], dtype=np.float64))
            y = h @ x
            y *= alpha
            if uniform:
                y += (1-alpha + alpha*dm) / n
            elif same_pq:
                y += (1-alpha + alpha*dm) * pp
            else:
                y += (1-alpha) * pp + (alpha*dm) * qq
            delta = float(np.sum(np.abs(y-x), dtype=np.float64))
            x = y
            if history:
                record[iterations-1] = delta
            if delta <= threshold:
                break
        elapsed = perf_counter() - start
        scores = x
    else:
        from .analysis import solve_linear
        start = perf_counter()
        scores = solve_linear(graph, alpha=alpha, personalization=p, dangling=q,
                              method=method, tol=tol, max_iter=max_iter)
        elapsed = perf_counter() - start
        # A linear solver does not have a comparable power iteration count.
        iterations = 0
    # Do not silently renormalize a failed iterate. Test the actual returned x.
    x64 = scores.astype(np.float64, copy=False)
    mapped = alpha * (graph.incoming @ x64 + q * x64[graph.dangling].sum()) + (1-alpha)*p
    residual = float(np.sum(np.abs(mapped-x64)))
    bound = residual / (1-alpha)
    converged = bool(np.isfinite(bound) and bound <= tol and np.all(x64 >= 0))
    scores.flags.writeable = False
    hist = record[:iterations].copy() if history else None
    result = PageRankResult(scores, iterations, converged, residual, bound, float(delta), elapsed,
                            alpha, tol, method, str(dtype), threads, hist, partition)
    if not converged and raise_on_failure:
        raise ConvergenceError(result)
    return result


def alpha_sweep(graph, alphas, *, warm_start=True, **kwargs):
    """Solve in supplied alpha order, optionally recycling the preceding vector."""
    if "alpha" in kwargs:
        raise ValueError("pass alpha values through alphas")
    x = kwargs.pop("x0", None)
    results = []
    for alpha in alphas:
        result = pagerank(graph, alpha=alpha, x0=x, **kwargs)
        results.append(result)
        if warm_start:
            x = result.scores
    return results


def personalized_batch(graph, personalizations, *, dangling=None, **kwargs):
    """Bounded-memory batch convenience API, with one query per row.

    Reuses prepared graph storage, but solves queries sequentially. It does not
    construct an n-by-n personalized PageRank matrix or claim block-SpMM speed.
    """
    ps = np.asarray(personalizations)
    if ps.ndim != 2 or ps.shape[1] != graph.n:
        raise ValueError(f"personalizations must have shape (queries, {graph.n})")
    return [pagerank(graph, personalization=p, dangling=dangling, **kwargs) for p in ps]
