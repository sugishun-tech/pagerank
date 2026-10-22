"""Linear algebra and sensitivity tools for the column-vector PageRank model.

Let ``H`` be the graph's row-normalized adjacency matrix (zero dangling rows),
``d`` its dangling indicator, ``p`` the teleport distribution, and ``q`` the
dangling distribution.  We use ``T = H.T + q d.T`` and solve
``(I - alpha T) x = (1 - alpha) p``.  The default is ``q = p``.

Residual error estimates are mathematical bounds in exact arithmetic.  Their
floating-point evaluations are diagnostics, not interval-arithmetic proofs.
All analytical routines use float64, independently of the power solver dtype.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import LinearOperator, gmres, splu

if TYPE_CHECKING:
    from .graph import Graph

__all__ = [
    "google_operator", "residual", "solve_linear", "alpha_derivative",
    "personalization_derivative", "edge_derivative", "edge_gradient",
    "exact_pagerank", "spectrum", "rank_scores", "compare_scores",
    "iteration_bound", "neumann_series",
]


def _alpha(value: float) -> float:
    a = float(value)
    if not math.isfinite(a) or not 0.0 <= a < 1.0:
        raise ValueError("alpha must be finite and satisfy 0 <= alpha < 1")
    return a


def _vector(value: Any, n: int, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({n},)")
    return result


def _probability(value: Any, n: int, name: str) -> np.ndarray:
    if n < 1:
        raise ValueError("a nonempty graph is required")
    if value is None:
        return np.full(n, 1.0 / n)
    result = _vector(value, n, name)
    if np.any(result < 0) or not np.any(result > 0):
        raise ValueError(f"{name} must be nonnegative and have positive mass")
    # Scaling first avoids overflow when finite weights have an infinite sum.
    result = result / np.max(result)
    return result / np.sum(result)


def _model(graph: Graph, alpha: float, personalization: Any, dangling: Any):
    a = _alpha(alpha)
    p = _probability(personalization, graph.n, "personalization")
    q = p if dangling is None else _probability(dangling, graph.n, "dangling")
    h = graph.transition.astype(np.float64, copy=False)
    incoming = graph.incoming.astype(np.float64, copy=False)
    d = np.asarray(graph.dangling, dtype=bool)
    return a, p, q, h, incoming, d


def _transition_product(incoming, q, d, values):
    if values.ndim == 1:
        return incoming @ values + q * np.sum(values[d])
    return incoming @ values + q[:, None] * np.sum(values[d], axis=0)


def google_operator(
    graph: Graph, alpha: float = 0.85, personalization=None, dangling=None,
) -> LinearOperator:
    """Return the column-stochastic Google matrix as a matrix-free operator.

    Both ``G @ v`` and ``G.T @ v`` work for arbitrary real vectors, including
    vectors whose sum is not one.  Teleportation is ``(1-alpha)*p*sum(v)``;
    using an affine PageRank step here would not be a linear operator.
    ``matmat`` is implemented for multiple right-hand sides.
    """
    a, p, q, h, incoming, d = _model(graph, alpha, personalization, dangling)
    n = graph.n

    def matvec(v):
        v = np.asarray(v).reshape(n)
        return a * _transition_product(incoming, q, d, v) + (1-a) * p * np.sum(v)

    def rmatvec(v):
        v = np.asarray(v).reshape(n)
        return a * (h @ v + d * np.dot(q, v)) + (1-a) * np.dot(p, v)

    def matmat(v):
        v = np.asarray(v)
        return a * _transition_product(incoming, q, d, v) + (1-a) * p[:, None] * np.sum(v, axis=0)

    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec,
                          matmat=matmat, dtype=np.float64)


def residual(
    graph: Graph, scores, alpha: float = 0.85, personalization=None, dangling=None,
) -> dict[str, float | bool]:
    """Inspect a candidate and evaluate ``||F(x)-x||_1 / (1-alpha)``.

    A candidate need not already be a probability vector.  The affine residual
    bounds its distance to the unique PageRank vector in exact arithmetic.
    ``mass_error`` and ``nonnegative`` report additional probability checks.
    """
    a, p, q, _, incoming, d = _model(graph, alpha, personalization, dangling)
    x = _vector(scores, graph.n, "scores")
    r = a * _transition_product(incoming, q, d, x) + (1-a) * p - x
    norm = float(np.linalg.norm(r, ord=1))
    mass = float(np.sum(x))
    return {
        "residual_l1": norm,
        "error_bound_l1": norm / (1-a),
        "mass": mass,
        "mass_error": abs(mass-1.0),
        "min_score": float(np.min(x)),
        "nonnegative": bool(np.all(x >= 0)),
    }


class _LinearSystem:
    """Sparse LU with the dangling rank-one correction kept out of the matrix."""

    def __init__(self, incoming, q, d, a):
        self.a = a
        self.q = q
        self.d = d
        self.incoming = incoming
        self.n = incoming.shape[0]
        base = sparse.eye(self.n, format="csc") - a * incoming.tocsc()
        self.lu = splu(base)
        self.z = self.lu.solve(q)
        self.denominator = 1.0 - a * np.sum(self.z[d])
        if not self.denominator > 0 or not np.isfinite(self.denominator):
            raise ArithmeticError("dangling rank-one correction is numerically singular")

    def solve(self, rhs):
        y = self.lu.solve(np.asarray(rhs, dtype=np.float64))
        if y.ndim == 1:
            return y + self.z * (self.a * np.sum(y[self.d]) / self.denominator)
        return y + self.z[:, None] * (self.a * np.sum(y[self.d], axis=0) / self.denominator)

    def solve_transpose(self, rhs):
        y = self.lu.solve(np.asarray(rhs, dtype=np.float64), trans="T")
        z = self.lu.solve(self.d.astype(np.float64), trans="T")
        return y + z * (self.a * np.dot(self.q, y) / self.denominator)

    def apply(self, values):
        return values - self.a * _transition_product(self.incoming, self.q, self.d, values)


def solve_linear(
    graph: Graph, alpha: float = 0.85, personalization=None, dangling=None,
    *, method: str = "direct", tol: float = 1e-10, max_iter: int = 1000,
    restart: int | None = None,
) -> np.ndarray:
    """Solve PageRank with sparse LU or matrix-free GMRES, in float64.

    ``tol`` targets the a posteriori L1 error estimate.  The direct method
    factors ``I-alpha*H.T`` and applies Sherman--Morrison for dangling nodes.
    LU fill-in can still be large; use GMRES or the power solver for large
    graphs.  GMRES ``max_iter`` counts restart cycles, as in SciPy.  A failed
    iteration or an unattainable residual target raises ``RuntimeError``.
    """
    a, p, q, _, incoming, d = _model(graph, alpha, personalization, dangling)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if method not in {"direct", "gmres"}:
        raise ValueError("method must be 'direct' or 'gmres'")
    if a == 0:
        return p.copy()
    b = (1-a) * p
    if method == "direct":
        system = _LinearSystem(incoming, q, d, a)
        x = system.solve(b)
        # Refinement helps when alpha is close to one, without hiding the
        # residual behind arbitrary renormalization or clipping.
        for _ in range(3):
            correction = b - system.apply(x)
            if np.linalg.norm(correction, ord=1) <= tol * (1-a):
                break
            x = x + system.solve(correction)
    else:
        def matvec(v):
            v = np.asarray(v).reshape(graph.n)
            return v - a * _transition_product(incoming, q, d, v)

        operator = LinearOperator((graph.n, graph.n), matvec=matvec, dtype=np.float64)
        # ||r||_1 <= sqrt(n)||r||_2, so SciPy's 2-norm target is sufficient.
        x, info = gmres(operator, b, rtol=0.0,
                        atol=tol * (1-a) / (2 * math.sqrt(graph.n)),
                        restart=restart, maxiter=max_iter)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
    report = residual(graph, x, a, p, q)
    if not np.all(np.isfinite(x)) or report["error_bound_l1"] > tol:
        raise RuntimeError(
            f"linear solve did not meet the L1 error target: "
            f"bound={report['error_bound_l1']:.3g}, tol={tol:.3g}"
        )
    return np.asarray(x)


def _sensitivity_context(graph, alpha, personalization, dangling, scores):
    a, p, q, h, incoming, d = _model(graph, alpha, personalization, dangling)
    system = _LinearSystem(incoming, q, d, a)
    x = system.solve((1-a) * p) if scores is None else _vector(scores, graph.n, "scores")
    return a, p, q, h, incoming, d, system, x


def alpha_derivative(
    graph: Graph, alpha: float = 0.85, personalization=None, dangling=None,
    *, scores=None,
) -> np.ndarray:
    """Compute ``dx/dalpha = (I-alpha*T)^-1 (T*x-p)``.

    ``p`` and ``q`` are held fixed.  With ``dangling=None``, ``q=p`` is also
    fixed because alpha alone changes.  An optional approximate ``scores``
    vector avoids the initial solve; its error propagates to the derivative.
    """
    _, p, q, _, incoming, d, system, x = _sensitivity_context(
        graph, alpha, personalization, dangling, scores)
    return system.solve(_transition_product(incoming, q, d, x) - p)


def personalization_derivative(
    graph: Graph, direction, alpha: float = 0.85, personalization=None,
    dangling=None, *, scores=None,
) -> np.ndarray:
    """Differentiate along the probability-simplex path ``p(t)=p+t*h``.

    ``direction=h`` must sum to zero.  No normalization derivative is silently
    inserted: ``p`` is normalized first and ``h`` refers to that distribution.
    When ``dangling=None``, the default ``q(t)=p(t)`` changes as well and the
    right-hand side is ``[(1-alpha)+alpha*d.T*x]*h``.  With explicit ``q``, it
    is ``(1-alpha)*h``.  At the simplex boundary, only feasible one-sided
    perturbations are probabilistic; the formula remains an affine derivative.
    """
    h_direction = _vector(direction, graph.n, "direction")
    if abs(np.sum(h_direction)) > 1e-12 * max(1.0, np.linalg.norm(h_direction, ord=1)):
        raise ValueError("direction must sum to zero on the probability simplex")
    a, _, _, _, _, d, system, x = _sensitivity_context(
        graph, alpha, personalization, dangling, scores)
    factor = 1-a + (a * np.sum(x[d]) if dangling is None else 0.0)
    return system.solve(factor * h_direction)


def _node(value: int, n: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not 0 <= value < n:
        raise ValueError(f"{name} must be an integer in [0, {n})")
    return int(value)


def edge_derivative(
    graph: Graph, source: int, target: int, alpha: float = 0.85,
    personalization=None, dangling=None, *, scores=None,
) -> np.ndarray:
    """Differentiate scores with respect to raw aggregate edge weight w_ij.

    All other raw weights remain fixed; row normalization is differentiated.
    For a nondangling source i with strength s_i, the right-hand side is
    ``alpha*x_i*(e_j-H[i,:])/s_i``.  An absent edge has a right derivative.
    A dangling source is rejected because its transition rule changes
    discontinuously as the first positive outgoing edge is introduced.
    Parallel input edges are represented by their aggregate weight.
    """
    i = _node(source, graph.n, "source")
    j = _node(target, graph.n, "target")
    a, _, _, h, _, d, system, x = _sensitivity_context(
        graph, alpha, personalization, dangling, scores)
    if d[i]:
        raise ValueError("edge derivatives require a nondangling source")
    row = graph.adjacency.getrow(i)
    strength = float(np.sum(row.data))
    rhs = np.zeros(graph.n)
    hrow = h.getrow(i)
    rhs[hrow.indices] = -hrow.data
    rhs[j] += 1.0
    # Divide the completed numerator, not its reciprocal: 1/s can overflow
    # for subnormal weights even when the derivative is exactly zero.
    with np.errstate(over="ignore", invalid="ignore"):
        rhs = (rhs * (a * x[i])) / strength
    if not np.all(np.isfinite(rhs)):
        raise ArithmeticError("edge derivative exceeds float64 range; rescale outgoing weights")
    derivative = system.solve(rhs)
    if not np.all(np.isfinite(derivative)):
        raise ArithmeticError("edge derivative solution exceeds float64 range; rescale outgoing weights")
    return derivative


def edge_gradient(
    graph: Graph, objective, alpha: float = 0.85, personalization=None,
    dangling=None, *, scores=None,
) -> sparse.csr_matrix:
    """Return derivatives of ``objective.T @ x`` for every stored edge.

    One adjoint solve computes all edge derivatives.  The returned CSR has
    the adjacency's stored support; an unstored entry means "not evaluated",
    rather than a claim that an absent edge has zero right derivative.  Use
    :func:`edge_derivative` to examine an absent edge explicitly.
    """
    c = _vector(objective, graph.n, "objective")
    a, _, _, h, _, d, system, x = _sensitivity_context(
        graph, alpha, personalization, dangling, scores)
    adjoint = system.solve_transpose(c)
    row_mean = h @ adjoint
    result = graph.adjacency.astype(np.float64, copy=True)
    for i in range(graph.n):
        start, stop = result.indptr[i:i+2]
        if start == stop or d[i]:
            continue
        strength = float(np.sum(result.data[start:stop]))
        with np.errstate(over="ignore", invalid="ignore"):
            result.data[start:stop] = (
                (a * x[i] * (adjoint[result.indices[start:stop]] - row_mean[i])) / strength
            )
    if not np.all(np.isfinite(result.data)):
        raise ArithmeticError("edge gradient exceeds float64 range; rescale outgoing weights")
    return result


def exact_pagerank(
    graph: Graph, alpha="17/20", personalization=None, dangling=None,
    *, max_n: int = 32,
):
    """Solve a small graph exactly with SymPy rational arithmetic.

    Graph weights are interpreted as the exact decimals represented by
    ``str(float(weight))``.  Thus integer-weight examples are exactly their
    mathematical graphs, while original fractions rounded during graph
    construction cannot be recovered; integer weights must also be exactly
    representable after aggregation in float64. ``alpha`` and distributions may use
    rational strings such as ``'1/3'``.  Install the ``analysis`` extra.
    A size guard prevents accidental dense symbolic solves on large graphs.
    """
    if graph.n < 1 or graph.n > max_n:
        raise ValueError(f"exact_pagerank requires 1 <= n <= max_n ({max_n})")
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError("exact_pagerank requires SymPy; install pagerank-lab[analysis]") from exc
    try:
        a = sp.Rational(str(alpha))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("alpha must be a rational number") from exc
    if a.is_finite is not True or a.is_real is not True or not 0 <= a < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    n = graph.n

    def distribution(value, name):
        if value is None:
            return sp.ones(n, 1) / n
        if len(value) != n:
            raise ValueError(f"{name} must have length {n}")
        try:
            values = [sp.Rational(str(v)) for v in value]
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{name} must contain rational numbers") from exc
        if any(v.is_finite is not True or v.is_real is not True for v in values):
            raise ValueError(f"{name} must contain finite rational numbers")
        if any(v < 0 for v in values) or sum(values) <= 0:
            raise ValueError(f"{name} must be nonnegative and have positive mass")
        return sp.Matrix(values) / sum(values)

    p = distribution(personalization, "personalization")
    q = p if dangling is None else distribution(dangling, "dangling")
    transition = sp.zeros(n)
    raw = graph.adjacency.tocsr()
    for i in range(n):
        start, stop = raw.indptr[i:i+2]
        values = [sp.Rational(str(float(v))) for v in raw.data[start:stop]]
        strength = sum(values)
        if strength:
            for j, weight in zip(raw.indices[start:stop], values):
                transition[int(j), i] = weight / strength
        else:
            transition[:, i] = q
    # LUsolve avoids explicitly forming an inverse; use exact arithmetic.
    x = (sp.eye(n) - a * transition).LUsolve((1-a) * p)
    return sp.ImmutableMatrix(x)


def spectrum(
    graph: Graph, alpha: float = 0.85, personalization=None, dangling=None,
    *, max_n: int = 512,
) -> dict[str, Any]:
    """Inspect all Google-matrix eigenvalues for a small graph.

    Results are sorted by decreasing magnitude.  The subdominant modulus is
    an asymptotic spectral diagnostic, not by itself a finite-iteration error
    bound for a nonnormal matrix.  Dense storage/time are O(n^2)/O(n^3), so
    ``max_n`` must be raised explicitly for larger experiments.
    """
    if graph.n > max_n:
        raise ValueError(f"spectrum requires n <= max_n ({max_n})")
    operator = google_operator(graph, alpha, personalization, dangling)
    eigenvalues = np.linalg.eigvals(operator @ np.eye(graph.n))
    order = np.argsort(-np.abs(eigenvalues), kind="stable")
    eigenvalues = eigenvalues[order]
    subdominant = float(abs(eigenvalues[1])) if graph.n > 1 else 0.0
    return {
        "eigenvalues": eigenvalues,
        "subdominant_modulus": subdominant,
        "spectral_gap": 1.0-subdominant,
    }


def rank_scores(scores, *, method: str = "average") -> np.ndarray:
    """Return descending ranks (best=1), with explicit treatment of ties."""
    if np.iscomplexobj(scores):
        raise ValueError("scores must be real")
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("scores must be a nonempty finite one-dimensional vector")
    if method not in {"average", "min", "max", "dense", "ordinal"}:
        raise ValueError("invalid tie method")
    return stats.rankdata(-x, method=method)


def compare_scores(x, y, *, top_k: int | None = None) -> dict[str, float]:
    """Compare vectors by L1/Linf distance, cosine, Spearman, and top-k overlap.

    No normalization is performed.  Undefined correlations (constant ranks)
    and zero-vector cosines are returned as NaN.  Top-k breaks score ties by
    ascending node index, so the selected sets always contain exactly k nodes.
    """
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError("x must be a nonempty finite one-dimensional vector")
    b = _vector(y, a.size, "y")
    if top_k is None:
        top_k = min(10, a.size)
    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or not 1 <= top_k <= a.size:
        raise ValueError("top_k must be an integer between 1 and the vector length")
    ranks_a = rank_scores(a)
    ranks_b = rank_scores(b)
    if np.ptp(ranks_a) == 0 or np.ptp(ranks_b) == 0:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    scale_a, scale_b = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    if scale_a and scale_b:
        unit_a, unit_b = a / scale_a, b / scale_b
        cosine = float(np.clip(np.dot(unit_a / np.linalg.norm(unit_a), unit_b / np.linalg.norm(unit_b)), -1, 1))
    else:
        cosine = float("nan")
    top_a = np.argsort(-a, kind="stable")[:top_k]
    top_b = np.argsort(-b, kind="stable")[:top_k]
    return {
        "l1": float(np.linalg.norm(a-b, ord=1)),
        "linf": float(np.linalg.norm(a-b, ord=np.inf)),
        "cosine": cosine,
        "spearman": spearman,
        "top_k_overlap": float(np.intersect1d(top_a, top_b).size / top_k),
    }


def iteration_bound(alpha: float, tol: float, *, initial_error_bound: float = 2.0) -> int:
    """Worst-case power steps from ``alpha**k * initial_error_bound <= tol``.

    Any two probability vectors have L1 distance at most two.  A smaller
    initial bound may be supplied when independently justified.  The returned
    value is an exact-arithmetic estimate and excludes floating-point floors.
    """
    a = _alpha(alpha)
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if not math.isfinite(initial_error_bound) or initial_error_bound < 0:
        raise ValueError("initial_error_bound must be nonnegative and finite")
    if tol >= initial_error_bound:
        return 0
    if a == 0:
        return 1
    return max(0, math.ceil((math.log(tol) - math.log(initial_error_bound)) / math.log(a)))


def neumann_series(
    graph: Graph, terms: int, alpha: float = 0.85, personalization=None, dangling=None,
) -> dict[str, Any]:
    """Return the first ``terms`` walk-length contributions and their sum.

    Column k is ``(1-alpha)*alpha**k*T**k*p``.  The omitted nonnegative tail
    has L1 mass exactly ``alpha**terms`` in exact arithmetic.  This is a
    decomposition, so the partial sum is deliberately not renormalized.
    Storage is O(n*terms); callers should choose small terms for large graphs.
    """
    if not isinstance(terms, (int, np.integer)) or isinstance(terms, bool) or terms < 1:
        raise ValueError("terms must be a positive integer")
    a, p, q, _, incoming, d = _model(graph, alpha, personalization, dangling)
    contributions = np.empty((graph.n, terms), dtype=np.float64)
    value = (1-a) * p
    for k in range(terms):
        contributions[:, k] = value
        value = a * _transition_product(incoming, q, d, value)
    return {
        "contributions": contributions,
        "partial_sum": contributions.sum(axis=1),
        "tail_l1_bound": a ** terms,
    }
