"""Small graph: exact arithmetic, sensitivity, and the spectral picture."""
from pathlib import Path

import numpy as np

from pagerank_lab import pagerank
from pagerank_lab.analysis import (
    alpha_derivative,
    edge_derivative,
    exact_pagerank,
    neumann_series,
    personalization_derivative,
    spectrum,
)
from pagerank_lab.io import load_graph

graph = load_graph(Path(__file__).with_name("toy.csv"))
alpha = 0.85
result = pagerank(graph, alpha=alpha, tol=1e-12)

print("PageRank:", result.scores)
print("Exact rational solution:", exact_pagerank(graph, alpha="17/20"))
print("Derivative with respect to alpha:", alpha_derivative(graph, alpha=alpha))
print("Derivative with respect to weight w[0,1]:", edge_derivative(graph, 0, 1, alpha=alpha))

# Infinitesimally transfer teleport mass from node 1 to node 0.
direction = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
print("Teleport directional derivative:", personalization_derivative(graph, direction, alpha=alpha))
print("Google matrix eigenvalues:", spectrum(graph, alpha=alpha)["eigenvalues"])
series = neumann_series(graph, terms=30, alpha=alpha)
print("Neumann truncation L1 bound:", series["tail_l1_bound"])

