"""Seeded synthetic graph families for controlled numerical experiments.

``m`` is the number of edge samples, not the number of distinct final edges:
parallel samples are summed. No model is advertised as a real web crawl.
"""
from __future__ import annotations

import numbers

import numpy as np

from .graph import Graph


def _parameters(n: int, m: int, dangling_fraction: float) -> tuple[int, int, int]:
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(m, bool) or not isinstance(m, numbers.Integral) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    if not np.isfinite(dangling_fraction) or not 0 <= dangling_fraction <= 1:
        raise ValueError("dangling_fraction must lie in [0,1]")
    n, m = int(n), int(m)
    active = n - int(n * dangling_fraction)
    if active == 0 and m:
        raise ValueError("positive m is incompatible with all nodes being forced dangling")
    return n, m, active


def _weights(rng: np.random.Generator, m: int, weighted: bool) -> np.ndarray | None:
    return rng.uniform(0.1, 2.0, size=m) if weighted else None


def uniform_graph(
    n: int, m: int, *, seed: int = 0, dangling_fraction: float = 0.05,
    weighted: bool = False,
) -> Graph:
    """Draw endpoints independently, reserving a suffix of dangling nodes."""
    n, m, active = _parameters(n, m, dangling_fraction)
    rng = np.random.default_rng(seed)
    source = rng.integers(max(active, 1), size=m)
    target = rng.integers(n, size=m)
    return Graph.from_edges(source, target, weights=_weights(rng, m, weighted), n=n)


def skewed_graph(
    n: int, m: int, *, seed: int = 0, exponent: float = 1.2,
    dangling_fraction: float = 0.05, weighted: bool = False,
) -> Graph:
    """Zipf-like target popularity with uniformly sampled active sources.

    This is a static synthetic heavy-tail model, not preferential attachment
    and not a claim that the realized degree distribution is exactly a power law.
    """
    n, m, active = _parameters(n, m, dangling_fraction)
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent must be finite and positive")
    rng = np.random.default_rng(seed)
    popularity = np.exp(-exponent * np.log(np.arange(1, n + 1, dtype=np.float64)))
    popularity /= popularity.sum()
    # Permuting popularity prevents node number itself being a ranking cue.
    popularity = rng.permutation(popularity)
    source = rng.integers(max(active, 1), size=m)
    target = rng.choice(n, size=m, p=popularity)
    return Graph.from_edges(source, target, weights=_weights(rng, m, weighted), n=n)


def block_graph(
    n: int, m: int, *, seed: int = 0, blocks: int = 4,
    within_probability: float = 0.9, dangling_fraction: float = 0.05,
    weighted: bool = False,
) -> Graph:
    """Draw targets within the source block or globally.

    With probability within_probability a target comes from the same block;
    otherwise it comes from all nodes, so the latter branch may also stay within.
    Block sizes differ by at most one node.
    """
    n, m, active = _parameters(n, m, dangling_fraction)
    if isinstance(blocks, bool) or not isinstance(blocks, numbers.Integral) or not 1 <= blocks <= n:
        raise ValueError("blocks must be an integer in [1,n]")
    if not np.isfinite(within_probability) or not 0 <= within_probability <= 1:
        raise ValueError("within_probability must lie in [0,1]")
    rng = np.random.default_rng(seed)
    source = rng.integers(max(active, 1), size=m)
    target = rng.integers(n, size=m)
    boundaries = np.linspace(0, n, int(blocks) + 1, dtype=np.int64)
    memberships = np.searchsorted(boundaries[1:], source, side="right")
    local = rng.random(m) < within_probability
    low = boundaries[memberships[local]]
    high = boundaries[memberships[local] + 1]
    target[local] = rng.integers(low, high)
    return Graph.from_edges(source, target, weights=_weights(rng, m, weighted), n=n)

