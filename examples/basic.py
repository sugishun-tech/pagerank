"""Run from the repository root after installing the package."""
from pathlib import Path

import numpy as np

from pagerank_lab import pagerank
from pagerank_lab.io import load_graph, load_vector

directory = Path(__file__).resolve().parent
graph = load_graph(directory / "toy.csv")
personalization = load_vector(directory / "personalization.json", n=graph.n)

fast = pagerank(graph, personalization=personalization, tol=1e-11, history=True)
reference = pagerank(graph, personalization=personalization, tol=1e-11, method="scipy")

print("scores:", fast.scores)
print("iterations:", fast.iterations)
print("residual L1:", fast.residual_l1)
print("error estimate L1:", fast.error_bound_l1)
print("Cython/SciPy difference L1:", np.linalg.norm(fast.scores - reference.scores, 1))
print("Both converged:", fast.converged and reference.converged)

