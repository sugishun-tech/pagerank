"""Sparse PageRank computation and mathematical diagnostics."""
from .graph import Graph
from .core import (PageRankResult, ConvergenceError, pagerank, alpha_sweep,
                   personalized_batch, build_info)

__version__ = "0.1.0"

_ANALYSIS = {
    "google_operator", "residual", "solve_linear", "alpha_derivative",
    "personalization_derivative", "edge_derivative", "exact_pagerank",
    "spectrum", "rank_scores", "compare_scores", "edge_gradient", "iteration_bound", "neumann_series",
}

def __getattr__(name):
    if name in _ANALYSIS:
        from . import analysis
        return getattr(analysis, name)
    raise AttributeError(name)

__all__ = ["Graph", "PageRankResult", "ConvergenceError", "pagerank", "alpha_sweep",
           "personalized_batch", "build_info", *_ANALYSIS]
