"""Independent mathematical fixtures, adjoint identities, and finite differences."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pagerank_lab.graph import Graph
from pagerank_lab.analysis import (
    alpha_derivative, compare_scores, edge_derivative, edge_gradient,
    exact_pagerank, google_operator, iteration_bound, neumann_series,
    personalization_derivative, rank_scores, residual, solve_linear, spectrum,
)


@pytest.fixture
def weighted_graph():
    # Node 3 is a sink; node 4 is isolated.  Include nonuniform outgoing weights.
    matrix = np.array([
        [0, 2, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=float)
    return Graph.from_scipy(sparse.csr_matrix(matrix))


@pytest.fixture
def distributions():
    return np.array([.1, .2, .25, .3, .15]), np.array([.5, 0, .1, .3, .1])


def _dense_reference(graph, a, p, q):
    # Deliberately construct the full small Markov matrix independently of
    # graph.transition and graph.incoming.
    raw = graph.adjacency.toarray().astype(float)
    p = np.ones(graph.n) / graph.n if p is None else np.asarray(p) / np.sum(p)
    q = p if q is None else np.asarray(q) / np.sum(q)
    for i, row in enumerate(raw):
        raw[i] = row / row.sum() if row.sum() else q
    return np.linalg.solve(np.eye(graph.n) - a * raw.T, (1-a) * p)


@pytest.mark.parametrize("method", ["direct", "gmres"])
@pytest.mark.parametrize("alpha", [0.0, 0.85, 0.99])
def test_linear_solvers_against_independent_dense_reference(weighted_graph, distributions, method, alpha):
    p, q = distributions
    x = solve_linear(weighted_graph, alpha, p, q, method=method, tol=1e-10)
    assert_allclose(x, _dense_reference(weighted_graph, alpha, p, q), atol=1e-11, rtol=0)
    assert abs(x.sum()-1) < 1e-11


def test_google_operator_is_linear_and_adjoint(weighted_graph, distributions):
    p, q = distributions
    operator = google_operator(weighted_graph, .8, p, q)
    u = np.arange(5.) - 2
    v = np.array([2., -4., 3., 1., .5])
    dense = operator @ np.eye(5)
    assert_allclose(dense.sum(axis=0), 1)
    assert_allclose(operator @ (2*u-3*v), 2*(operator @ u)-3*(operator @ v))
    assert_allclose(operator @ u, dense @ u)
    assert_allclose(operator.T @ v, dense.T @ v)
    assert_allclose(np.dot(v, operator @ u), np.dot(operator.T @ v, u))
    assert_allclose(operator @ np.column_stack((u, v)), dense @ np.column_stack((u, v)))


def test_residual_bound_applies_to_nonunit_mass_candidate(weighted_graph, distributions):
    p, q = distributions
    exact = _dense_reference(weighted_graph, .9, p, q)
    candidate = exact + np.array([.005, -.01, 0, .02, 0])
    info = residual(weighted_graph, candidate, .9, p, q)
    assert np.linalg.norm(candidate-exact, 1) <= info["error_bound_l1"] + 1e-14
    assert_allclose(info["mass_error"], .015)
    assert info["nonnegative"]


def test_hand_solved_two_node_graph_and_alpha_derivative():
    graph = Graph.from_scipy(sparse.csr_matrix([[0., 1.], [0., 0.]]))
    for alpha in [0., .5, .85]:
        expected = np.array([1., 1+alpha]) / (2+alpha)
        assert_allclose(solve_linear(graph, alpha), expected, atol=1e-14)
        assert_allclose(alpha_derivative(graph, alpha), [-1/(2+alpha)**2, 1/(2+alpha)**2], atol=1e-14)


@pytest.mark.parametrize("method", ["direct", "gmres"])
def test_all_dangling_with_distinct_distributions(method):
    graph = Graph.from_scipy(sparse.csr_matrix((3, 3)))
    p = np.array([1., 0., 0.])
    q = np.array([0., .25, .75])
    assert_allclose(solve_linear(graph, .8, p, q, method=method), .2*p+.8*q, atol=1e-14)


def test_alpha_derivative_finite_difference(weighted_graph, distributions):
    p, q = distributions
    epsilon = 1e-5
    finite_difference = (solve_linear(weighted_graph, .85+epsilon, p, q) -
                         solve_linear(weighted_graph, .85-epsilon, p, q)) / (2*epsilon)
    derivative = alpha_derivative(weighted_graph, .85, p, q)
    assert_allclose(derivative, finite_difference, rtol=2e-8, atol=1e-10)
    assert abs(derivative.sum()) < 1e-13


@pytest.mark.parametrize("default_dangling", [True, False])
def test_personalization_derivative_includes_default_dangling_dependence(weighted_graph, distributions, default_dangling):
    p, q = distributions
    if default_dangling:
        q = None
    direction = np.array([.2, -.3, .1, .15, -.15])
    epsilon = 1e-5
    finite_difference = (solve_linear(weighted_graph, .85, p+epsilon*direction, q) -
                         solve_linear(weighted_graph, .85, p-epsilon*direction, q)) / (2*epsilon)
    derivative = personalization_derivative(weighted_graph, direction, .85, p, q)
    assert_allclose(derivative, finite_difference, rtol=2e-8, atol=1e-10)
    assert abs(derivative.sum()) < 1e-13


@pytest.mark.parametrize("target", [1, 4])
def test_edge_derivative_existing_and_absent_edge(weighted_graph, distributions, target):
    p, q = distributions
    raw = weighted_graph.adjacency.toarray()
    epsilon = 1e-6
    plus = raw.copy()
    plus[0, target] += epsilon
    if raw[0, target] > 0:
        minus = raw.copy()
        minus[0, target] -= epsilon
        numerical = (solve_linear(Graph.from_scipy(sparse.csr_matrix(plus)), .85, p, q) -
                     solve_linear(Graph.from_scipy(sparse.csr_matrix(minus)), .85, p, q)) / (2*epsilon)
        tolerance = 2e-9
    else:
        numerical = (solve_linear(Graph.from_scipy(sparse.csr_matrix(plus)), .85, p, q) -
                     solve_linear(weighted_graph, .85, p, q)) / epsilon
        tolerance = 2e-8
    derivative = edge_derivative(weighted_graph, 0, target, .85, p, q)
    assert_allclose(derivative, numerical, rtol=2e-6, atol=tolerance)
    assert abs(derivative.sum()) < 1e-13


def test_edge_gradient_adjoint_and_row_scaling_invariance(weighted_graph, distributions):
    p, q = distributions
    objective = np.array([1., -.5, 3., 0., 1.])
    gradient = edge_gradient(weighted_graph, objective, .85, p, q)
    for i, j in zip(*weighted_graph.adjacency.nonzero()):
        derivative = edge_derivative(weighted_graph, i, j, .85, p, q)
        assert_allclose(gradient[i, j], objective @ derivative, atol=1e-14)
    # Rescaling all weights of any row cannot change normalized transitions.
    assert_allclose(np.asarray(gradient.multiply(weighted_graph.adjacency).sum(axis=1)).ravel(), 0, atol=1e-14)


def test_subnormal_single_edge_has_zero_weight_derivative():
    graph = Graph.from_scipy(sparse.csr_matrix([[0., 1e-320], [0., 0.]]))
    assert_allclose(edge_derivative(graph, 0, 1), 0, atol=0)
    assert_allclose(edge_gradient(graph, [1., 0.]).data, 0, atol=0)


def test_exact_pagerank_rational_hand_solution():
    sympy = pytest.importorskip("sympy")
    graph = Graph.from_scipy(sparse.csr_matrix([[0., 1.], [0., 0.]]))
    exact = exact_pagerank(graph, alpha="1/2")
    assert exact == sympy.ImmutableMatrix([sympy.Rational(2, 5), sympy.Rational(3, 5)])
    assert sum(exact) == 1
    with pytest.raises(ValueError):
        exact_pagerank(graph, alpha="1/0")
    with pytest.raises(ValueError):
        exact_pagerank(graph, personalization=["1/0", "1"])


def test_exact_personalized_matches_dense(weighted_graph):
    pytest.importorskip("sympy")
    exact = exact_pagerank(weighted_graph, "2/3", personalization=[1, 2, 3, 2, 1],
                          dangling=[0, 1, 0, 1, 0])
    numeric = np.array(exact, dtype=float).ravel()
    assert_allclose(numeric, _dense_reference(weighted_graph, 2/3, [1, 2, 3, 2, 1], [0, 1, 0, 1, 0]), atol=1e-14)


def test_spectrum_cycle_known_eigenvalues():
    graph = Graph.from_scipy(sparse.csr_matrix([[0., 1.], [1., 0.]]))
    result = spectrum(graph, alpha=.75)
    assert_allclose(result["eigenvalues"], [1., -.75], atol=1e-14)
    assert_allclose(result["spectral_gap"], .25)
    with pytest.raises(ValueError, match="max_n"):
        spectrum(graph, max_n=1)


def test_neumann_nonnegative_tail_is_exact_mass(weighted_graph, distributions):
    p, q = distributions
    x = solve_linear(weighted_graph, .85, p, q)
    result = neumann_series(weighted_graph, 25, .85, p, q)
    assert np.all(result["contributions"] >= 0)
    assert_allclose(result["contributions"].sum(axis=0), (1-.85)*.85**np.arange(25), atol=1e-14)
    assert_allclose(np.linalg.norm(x-result["partial_sum"], 1), .85**25, atol=1e-14)
    assert_allclose(result["tail_l1_bound"], .85**25)


def test_ranking_ties_and_metrics():
    scores = np.array([.4, .4, .2])
    assert_allclose(rank_scores(scores), [1.5, 1.5, 3])
    assert_allclose(rank_scores(scores, method="dense"), [1, 1, 2])
    assert_allclose(rank_scores(scores, method="ordinal"), [1, 2, 3])
    result = compare_scores(scores, scores, top_k=1)
    assert result["l1"] == 0
    assert result["top_k_overlap"] == 1
    assert_allclose(result["spearman"], 1)
    assert np.isnan(compare_scores([1, 1], [2, 2])["spearman"])
    assert_allclose(compare_scores([1e308, 5e307], [1e308, 5e307])["cosine"], 1)


def test_iteration_bound():
    n = iteration_bound(.85, 1e-10)
    assert 2*.85**n <= 1e-10
    assert 2*.85**(n-1) > 1e-10
    assert iteration_bound(0, 1e-10) == 1
    assert iteration_bound(.85, 2) == 0


@pytest.mark.parametrize("alpha", [-.1, 1., np.inf, np.nan])
def test_bad_alpha(weighted_graph, alpha):
    with pytest.raises(ValueError, match="alpha"):
        google_operator(weighted_graph, alpha)


def test_invalid_analysis_inputs(weighted_graph):
    with pytest.raises(ValueError, match="real"):
        solve_linear(weighted_graph, personalization=np.ones(5, dtype=complex))
    with pytest.raises(ValueError, match="sum to zero"):
        personalization_derivative(weighted_graph, np.ones(5))
    with pytest.raises(ValueError, match="nondangling"):
        edge_derivative(weighted_graph, 3, 0)
    with pytest.raises(ValueError, match="source"):
        edge_derivative(weighted_graph, -1, 0)
    with pytest.raises(ValueError, match="positive mass"):
        solve_linear(weighted_graph, personalization=np.zeros(5))
    with pytest.raises(ValueError, match="max_n"):
        exact_pagerank(weighted_graph, max_n=3)
    with pytest.raises(ValueError, match="tol"):
        solve_linear(weighted_graph, tol=0)
    with pytest.raises(ValueError, match="terms"):
        neumann_series(weighted_graph, terms=0)


def test_edge_derivative_solution_overflow_is_explicit():
    graph = Graph.from_edges([0, 1], [0, 1], [1e-308, 1.0], n=2)
    with pytest.raises(ArithmeticError, match="solution exceeds"):
        edge_derivative(graph, 0, 1, alpha=.85)
