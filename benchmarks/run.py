"""Compare implementations at an independently verified absolute L1 target.

Run from the repository root: ``python -m benchmarks.run --help``.
This is intentionally a source-checkout utility, not part of the installed API.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import sysconfig
import time
from typing import Any, Callable

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Config:
    sizes: tuple[int, ...] = (1000, 10000, 100000)
    families: tuple[str, ...] = ("uniform", "skewed", "block", "dangling")
    methods: tuple[str, ...] = ("cython", "scipy", "networkx", "igraph")
    threads: tuple[int, ...] = (1, 2, 4)
    alphas: tuple[float, ...] = (0.85,)
    degree: int = 16
    repeats: int = 7
    warmups: int = 2
    seed: int = 20261023
    tol: float = 1e-10
    max_iter: int = 10000
    weighted: bool = False
    personalization: str = "uniform"
    dangling_distribution: str = "personalization"
    python_max_n: int = 1000
    direct_max_n: int = 3000

    def validate(self) -> None:
        if not self.sizes or any(n < 1 for n in self.sizes):
            raise ValueError("sizes must contain positive integers")
        if not self.families or set(self.families) - {"uniform", "skewed", "block", "dangling", "cycle"}:
            raise ValueError("invalid or empty families")
        if not self.methods or set(self.methods) - {"python", "cython", "scipy", "direct", "gmres", "networkx", "igraph"}:
            raise ValueError("invalid or empty methods")
        if not self.threads or any(t < 1 for t in self.threads):
            raise ValueError("threads must contain positive integers")
        if not self.alphas or any(not math.isfinite(a) or not 0 <= a < 1 for a in self.alphas):
            raise ValueError("alphas must be finite and in [0,1)")
        if self.degree < 0 or self.repeats < 1 or self.warmups < 0 or self.max_iter < 1:
            raise ValueError("degree/warmups must be nonnegative; repeats/max_iter positive")
        if not math.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive and finite")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.python_max_n < 1:
            raise ValueError("python_max_n must be positive")
        if self.direct_max_n < 1:
            raise ValueError("direct_max_n must be positive")
        if self.personalization not in {"uniform", "random", "localized"}:
            raise ValueError("invalid personalization")
        if self.dangling_distribution not in {"personalization", "uniform", "random"}:
            raise ValueError("invalid dangling_distribution")


def make_adjacency(n: int, degree: int, family: str, seed: int, weighted: bool = False):
    """Return canonical directed CSR and generation metadata.

    ``skewed`` uses a fixed Zipf-like target distribution, NOT preferential
    attachment and NOT a claim of a power-law degree sequence. ``block`` has
    eight clusters with 98% of sampled targets within the source cluster.
    ``dangling`` suppresses every fifth source and retains those vertices.
    Duplicates add their weights. Self-loops are retained.
    """
    if n < 1 or degree < 0:
        raise ValueError("n must be positive and degree nonnegative")
    rng = np.random.default_rng(seed)
    if family == "cycle":
        src = np.arange(n, dtype=np.int64)
        dst = (src + 1) % n
    else:
        count = n * degree
        src = rng.integers(0, n, size=count, dtype=np.int64)
        if family == "skewed":
            probabilities = np.power(np.arange(1, n + 1, dtype=float), -1.1)
            probabilities /= probabilities.sum()
            dst = rng.choice(n, size=count, p=probabilities)
        elif family == "block":
            blocks = min(8, n)
            # Integer block boundaries permit non-divisible n without clipping.
            boundaries = np.linspace(0, n, blocks + 1, dtype=np.int64)
            group = np.searchsorted(boundaries[1:], src, side="right")
            dst = rng.integers(0, n, size=count, dtype=np.int64)
            local = rng.random(count) < 0.98
            widths = boundaries[group + 1] - boundaries[group]
            local_dst = boundaries[group] + (rng.random(count) * widths).astype(np.int64)
            dst[local] = local_dst[local]
        elif family in {"uniform", "dangling"}:
            dst = rng.integers(0, n, size=count, dtype=np.int64)
        else:
            raise ValueError(f"unknown family: {family}")
        if family == "dangling":
            keep = src % 5 != 0
            src, dst = src[keep], dst[keep]
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=src.size) if weighted else np.ones(src.size)
    adjacency = sparse.csr_matrix((weights, (src, dst)), shape=(n, n), dtype=np.float64)
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    adjacency.eliminate_zeros()
    meta = {
        "family": family, "n": n, "m": int(adjacency.nnz),
        "sampled_edges": int(src.size), "requested_degree": degree,
        "seed": seed, "weighted": weighted, "self_loops": "retained",
        "duplicate_edges": "summed", "generator": "numpy.default_rng(PCG64)",
    }
    h = hashlib.sha256()
    for arr in (adjacency.indptr.astype("<i8"), adjacency.indices.astype("<i8"), adjacency.data.astype("<f8")):
        h.update(arr.tobytes())
    meta["adjacency_sha256"] = h.hexdigest()
    return adjacency, meta


def networkx_tolerance(target: float, alpha: float, n: int) -> float:
    """Translate our absolute L1 bound to NX's ``delta < n*tol``."""
    if n < 1 or target <= 0 or not math.isfinite(target) or not 0 <= alpha < 1:
        raise ValueError("invalid target, alpha, or n")
    return target / n if alpha == 0 else target * (1.0 - alpha) / (alpha * n)


def independent_check(graph, scores, alpha: float, p, q, target: float) -> dict[str, Any]:
    """Evaluate the affine fixed-point equation using float64 SciPy.

    The residual/(1-alpha) certificate is an exact-arithmetic theorem evaluated
    in floating point here; it is not an outward-rounded interval certificate.
    No loose tolerance multiplier makes a failing output pass.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.shape != (graph.n,) or not np.isfinite(x).all():
        return {"valid": False, "reason": "shape or nonfinite output"}
    with np.errstate(over="ignore", invalid="ignore"):
        tx = alpha * (graph.incoming @ x + q * x[graph.dangling].sum()) + (1.0 - alpha) * p
        residual = float(np.abs(tx - x).sum())
        bound = residual / (1.0 - alpha)
        mass_error = abs(float(x.sum()) - 1.0)
    if not all(math.isfinite(value) for value in (residual, bound, mass_error)):
        return {"valid": False, "reason": "nonfinite residual or mass during independent check"}
    minimum = float(x.min())
    valid = bound <= target and minimum >= 0 and mass_error <= max(target, 32*np.finfo(float).eps)
    return {"valid": bool(valid), "residual_l1": residual,
            "error_bound_l1": bound, "mass_error": mass_error,
            "minimum_score": minimum,
            "reason": None if valid else "residual, mass, or nonnegativity check failed"}


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _peak_rss() -> int | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None


def environment() -> dict[str, Any]:
    affinity_before_build = sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
    versions = {}
    for package in ("numpy", "scipy", "Cython", "networkx", "igraph", "pagerank-lab", "sympy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), platform.processor())
    env = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "PAGERANK_OPENMP", "PAGERANK_NATIVE",
        "CC", "CFLAGS", "LDFLAGS")}
    try:
        from pagerank_lab import build_info
        build = build_info()
    except (ImportError, AttributeError):
        build = None
    return {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": model, "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "calling_thread_affinity_before_build_info": affinity_before_build,
        "affinity_semantics": "affinity is sched_getaffinity(0) of the calling thread after extension import; OpenMP binding may narrow it and it is not the union of worker-thread CPU places",
        "cgroup_cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": _read_text("/sys/fs/cgroup/memory.max"),
        "versions": versions, "environment": env,
        "python_compiler": platform.python_compiler(),
        "python_build_cc": sysconfig.get_config_var("CC"),
        "extension_build": build,
        "clock": vars(time.get_clock_info("perf_counter")),
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [s for s in samples if "wall_seconds" in s]
    times = [s["wall_seconds"] for s in successful]
    solver_times = [s["solver_seconds"] for s in successful if s.get("solver_seconds") is not None]
    if not times:
        return {"valid": False, "count": 0, "error": samples[0].get("error", "no samples") if samples else "no samples"}
    median = statistics.median(times)
    result = {
        "valid": all(s.get("valid", False) for s in samples), "count": len(times),
        "median_seconds": median, "min_seconds": min(times), "max_seconds": max(times),
        "q25_seconds": float(np.quantile(times, 0.25)), "q75_seconds": float(np.quantile(times, 0.75)),
        "mad_seconds": statistics.median(abs(t-median) for t in times),
        "solver_median_seconds": statistics.median(solver_times) if solver_times else None,
        "max_error_bound_l1": max((s["error_bound_l1"] for s in successful if "error_bound_l1" in s), default=None),
        "max_mass_error": max((s["mass_error"] for s in successful if "mass_error" in s), default=None),
        "iterations": sorted({s["iterations"] for s in successful if s.get("iterations") is not None}),
    }
    return result


def _timed(call: Callable[[], Any]):
    start = time.perf_counter_ns()
    value = call()
    return value, (time.perf_counter_ns()-start) * 1e-9


def python_power(incoming_rows, dangling_indices, alpha, p, q, tol, max_iter):
    """Plain Python gather/power reference on preprocessed numeric lists.

    Numerical input validation belongs to ``benchmark_graph``. This small
    reference deliberately keeps all iterative arithmetic in Python floats.
    """
    n = len(p)
    x = [1.0/n] * n
    started = time.perf_counter_ns()
    if alpha == 0:
        return np.asarray(p), {"iterations": 0, "solver_seconds": (time.perf_counter_ns()-started)*1e-9}
    threshold = tol*(1-alpha)/alpha
    for iteration in range(1,max_iter+1):
        mass = sum(x[i] for i in dangling_indices)
        y = []
        delta = 0.0
        for i,row in enumerate(incoming_rows):
            total = 0.0
            for j,weight in row:
                total += weight*x[j]
            value = alpha*(total + mass*q[i]) + (1-alpha)*p[i]
            y.append(value)
            delta += abs(value-x[i])
        x = y
        if delta <= threshold:
            return np.asarray(x), {"iterations": iteration, "solver_seconds": (time.perf_counter_ns()-started)*1e-9}
    raise RuntimeError(f"plain Python power iteration did not converge in {max_iter} steps")


def benchmark_graph(adjacency, graph_meta: dict, config: Config, alpha: float) -> dict[str, Any]:
    """Benchmark one graph; return raw samples and explicit timing scopes."""
    from pagerank_lab import Graph, pagerank

    graph, graph_seconds = _timed(lambda: Graph.from_scipy(adjacency))
    rng = np.random.default_rng(int(graph_meta["seed"]) + 1907)
    if config.personalization == "localized":
        p = np.zeros(graph.n)
        p[:max(1,graph.n//min(8,graph.n))] = 1.0
    else:
        p = np.ones(graph.n) if config.personalization == "uniform" else rng.random(graph.n)
    p /= p.sum()
    if config.dangling_distribution == "personalization":
        q = p.copy()
    else:
        q = np.ones(graph.n) if config.dangling_distribution == "uniform" else rng.random(graph.n)
        q /= q.sum()
    x0 = np.full(graph.n, 1.0/graph.n)
    preprocess = {"pagerank_graph_from_canonical_csr_seconds": graph_seconds}
    entries: dict[str, dict[str, Any]] = {}
    runners: dict[str, Callable[[], Any]] = {}

    for method in config.methods:
        if method == "python":
            if graph.n > config.python_max_n:
                entries[method] = {"method": method, "status": "skipped", "samples": [],
                    "reason": f"n exceeds configured python_max_n={config.python_max_n}"}
                continue
            def prepare_python():
                incoming = graph.incoming
                rows = [list(zip(incoming.indices[incoming.indptr[i]:incoming.indptr[i+1]].tolist(),
                                 incoming.data[incoming.indptr[i]:incoming.indptr[i+1]].tolist())) for i in range(graph.n)]
                return rows, np.flatnonzero(graph.dangling).tolist(), p.tolist(), q.tolist()
            (rows,di,plist,qlist), python_prepare_seconds = _timed(prepare_python)
            preprocess["python_lists_from_pagerank_graph_seconds"] = python_prepare_seconds
            entries[method] = {"method": method, "threads": 1, "samples": [],
                "preprocess_seconds": graph_seconds+python_prepare_seconds,
                "scope": "plain Python power on prepared numeric lists plus output ndarray and residual validation"}
            def python_call():
                scores, details = python_power(rows,di,alpha,plist,qlist,config.tol,config.max_iter)
                check = independent_check(graph,scores,alpha,p,q,config.tol)
                if not check["valid"]:
                    raise RuntimeError("plain Python result did not meet the common residual target")
                return scores, details
            runners[method] = python_call
        elif method in {"cython", "scipy", "direct", "gmres"}:
            if method == "direct" and graph.n > config.direct_max_n:
                entries[method] = {"method": method, "status": "skipped", "samples": [],
                    "reason": f"n exceeds configured direct_max_n={config.direct_max_n}; sparse factorization may fill in"}
                continue
            prepare_seconds = 0.0
            if method == "cython" and hasattr(graph, "prepare"):
                _, prepare_seconds = _timed(lambda: graph.prepare("float64"))
                preprocess["cython_prepare_seconds"] = prepare_seconds
            for threads in config.threads if method == "cython" else (1,):
                name = f"cython_t{threads}" if method == "cython" else "scipy"
                if method in {"direct", "gmres"}:
                    name = method
                partition_seconds = 0.0
                if method == "cython":
                    _, partition_seconds = _timed(lambda: graph.partition(threads, "balanced"))
                    preprocess[f"cython_partition_t{threads}_seconds"] = partition_seconds
                entries[name] = {"method": method, "threads": threads, "samples": [],
                    "preprocess_seconds": graph_seconds + prepare_seconds + partition_seconds,
                    "partition": "balanced" if method == "cython" else None,
                    "scope": "public API with prepared Graph, including input vectors, allocation and residual validation"}
                def ours(method=method, threads=threads):
                    result = pagerank(graph, alpha=alpha, personalization=p, dangling=q,
                        tol=config.tol, max_iter=config.max_iter, method=method, dtype="float64",
                        threads=threads, x0=x0, history=False)
                    return result.scores, {"iterations": int(result.iterations) if method not in {"direct", "gmres"} else None,
                        "solver_seconds": float(result.elapsed_seconds),
                        "reported_converged": bool(result.converged)}
                runners[name] = ours
        elif method == "networkx":
            try:
                import networkx as nx
            except ImportError:
                entries[method] = {"method": method, "status": "skipped", "reason": "networkx is not installed", "samples": []}
                continue
            def prepare_nx():
                ng = nx.from_scipy_sparse_array(adjacency, create_using=nx.DiGraph)
                return ng, dict(enumerate(p)), dict(enumerate(q)), dict(enumerate(x0))
            (ng, pdict, qdict, xdict), convert_seconds = _timed(prepare_nx)
            preprocess["networkx_from_canonical_csr_and_vectors_seconds"] = convert_seconds
            entries[method] = {"method": method, "threads": 1, "samples": [],
                "preprocess_seconds": convert_seconds,
                "networkx_tol": networkx_tolerance(config.tol, alpha, graph.n),
                "scope": "nx.pagerank public call on prepared DiGraph plus output ndarray; includes its internal sparse conversion"}
            def nx_call():
                result = nx.pagerank(ng, alpha=alpha, personalization=pdict, dangling=qdict,
                    nstart=xdict, max_iter=config.max_iter,
                    tol=networkx_tolerance(config.tol, alpha, graph.n), weight="weight")
                return np.fromiter((result[i] for i in range(graph.n)), dtype=float, count=graph.n), {"iterations": None, "solver_seconds": None}
            runners[method] = nx_call
        elif method == "igraph":
            try:
                import igraph as ig
            except ImportError:
                entries[method] = {"method": method, "status": "skipped", "reason": "igraph is not installed", "samples": []}
                continue
            if not np.array_equal(q, p):
                entries[method] = {"method": method, "status": "skipped", "reason": "PRPACK API uses reset distribution for dangling nodes; this case has q != p", "samples": []}
                continue
            def prepare_ig():
                coo = adjacency.tocoo()
                igg = ig.Graph(n=graph.n, edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=True)
                return igg, coo.data.tolist(), p.tolist()
            (igg, weights, reset), convert_seconds = _timed(prepare_ig)
            preprocess["igraph_from_canonical_csr_and_vectors_seconds"] = convert_seconds
            entries[method] = {"method": method, "threads": 1, "samples": [],
                "preprocess_seconds": convert_seconds,
                "scope": "igraph personalized_pagerank(PRPACK) public call plus output ndarray; its internal tolerance is not user-controlled"}
            def ig_call():
                result = igg.personalized_pagerank(directed=True, damping=alpha, reset=reset,
                    weights=weights, implementation="prpack")
                return np.asarray(result), {"iterations": None, "solver_seconds": None}
            runners[method] = ig_call

    # Warm every implementation first. Their discarded timings never enter a median.
    disabled = set()
    for name, call in runners.items():
        try:
            for _ in range(config.warmups):
                call()
        except Exception as exc:
            entries[name].update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            disabled.add(name)
    active = [name for name in runners if name not in disabled]
    order_rng = np.random.default_rng(int(graph_meta["seed"]) + 8051)
    orders = []
    # GC remains enabled: API allocations and cleanup are part of the user cost.
    for repeat in range(config.repeats):
        order = order_rng.permutation(active).tolist()
        orders.append(order)
        for name in order:
            try:
                (scores, details), wall = _timed(runners[name])
                check = independent_check(graph, scores, alpha, p, q, config.tol)
                sample = {"repeat": repeat, "wall_seconds": wall, **details, **check}
                if details.get("reported_converged") is False:
                    sample["valid"] = False
                    sample["reason"] = "solver reported nonconvergence"
                entries[name]["samples"].append(sample)
            except Exception as exc:
                entries[name]["samples"].append({"repeat": repeat, "valid": False,
                    "error": f"{type(exc).__name__}: {exc}"})
    for name, entry in entries.items():
        if "status" not in entry:
            entry["summary"] = summarize_samples(entry["samples"])
            entry["status"] = "ok" if entry["summary"]["valid"] else "failed"
            median = entry["summary"].get("median_seconds")
            if median is not None:
                # Explicit sum of separate measurements, NOT a fresh cold run.
                entry["summary"]["preprocess_plus_median_seconds"] = entry["preprocess_seconds"] + median
    return {
        "graph": {**graph_meta, "dangling_count": int(graph.dangling.sum())},
        "alpha": alpha, "tol_l1": config.tol, "dtype": "float64",
        "personalization": config.personalization, "dangling_distribution": config.dangling_distribution,
        "preprocessing": preprocess, "execution_orders": orders,
        "methods": entries, "process_peak_rss_bytes_cumulative": _peak_rss(),
    }


def run_suite(config: Config, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    config.validate()
    report = {
        "schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "environment": environment(),
        "timing_notes": [
            "All primary wall times include the full public solve API on a prepared graph and conversion of returned scores to ndarray.",
            "External graph construction and vector conversion are timed separately. Random graph generation is outside all solve times.",
            "NetworkX public pagerank internally rebuilds a sparse matrix each call; this is included and must not be called its iteration-only time.",
            "Our solver-only elapsed_seconds excludes public-wrapper and residual-check costs; only compare this subset to explicitly equivalent timings.",
            "Cython input arrays and each thread-count's balanced row partitions are prepared and timed before warmups; both costs enter preprocessing.",
            "Direct and GMRES linear solvers do not expose a comparable power-iteration count; their iterations field is null.",
            "preprocess_plus_median_seconds adds separately timed preparation to the warm-call median; it is NOT a directly measured cold end-to-end run.",
            "Every result must satisfy the same independently computed float64 residual/(1-alpha) target and nonnegative unit-mass checks.",
            "igraph PRPACK uses its own stopping criterion. Its run is comparable only when its independently measured residual passes the target.",
            "Method order is randomized per repetition with a recorded seed; all warmup runs are discarded; GC remains enabled.",
            "process_peak_rss_bytes_cumulative is the entire process high-water RSS including all methods and imports; it is NOT method-specific memory.",
            "Floating-point residual bounds implement an exact-arithmetic inequality; they are not interval-arithmetic certificates.",
        ], "cases": [],
    }
    for family_index, family in enumerate(config.families):
        for n in config.sizes:
            seed = config.seed + family_index*1000003 + n
            adjacency, meta = make_adjacency(n, config.degree, family, seed, config.weighted)
            for alpha in config.alphas:
                if progress:
                    progress(f"{family}: n={n:,} m={adjacency.nnz:,} alpha={alpha}")
                case = benchmark_graph(adjacency, meta, config, alpha)
                report["cases"].append(case)
                if progress:
                    for name, item in case["methods"].items():
                        median = item.get("summary", {}).get("median_seconds")
                        progress(f"  {name}: {item['status']}" + (f" {median*1e3:.3f} ms" if median is not None else f" {item.get('reason', '')}"))
            del adjacency
            gc.collect()
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def summary_rows(report: dict) -> list[dict]:
    rows = []
    for case in report["cases"]:
        for name, item in case["methods"].items():
            s = item.get("summary", {})
            rows.append({
                "family": case["graph"]["family"], "n": case["graph"]["n"], "m": case["graph"]["m"],
                "alpha": case["alpha"], "method": name, "status": item["status"],
                "repeats": s.get("count"), "median_seconds": s.get("median_seconds"),
                "min_seconds": s.get("min_seconds"), "max_seconds": s.get("max_seconds"),
                "q25_seconds": s.get("q25_seconds"), "q75_seconds": s.get("q75_seconds"),
                "mad_seconds": s.get("mad_seconds"), "solver_median_seconds": s.get("solver_median_seconds"),
                "preprocess_seconds": item.get("preprocess_seconds"),
                "preprocess_plus_median_seconds": s.get("preprocess_plus_median_seconds"),
                "max_error_bound_l1": s.get("max_error_bound_l1"),
                "max_mass_error": s.get("max_mass_error"),
                "iterations": ";".join(map(str, s.get("iterations", []))),
            })
    return rows


def write_report(report: dict, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = directory / "results.json", directory / "summary.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = summary_rows(report)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--families", nargs="+", default=["uniform", "skewed", "block", "dangling"])
    parser.add_argument("--methods", nargs="+", default=["cython", "scipy", "networkx", "igraph"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.85])
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--python-max-n", type=int, default=1000, help="skip optional pure-Python baseline above this size")
    parser.add_argument("--direct-max-n", type=int, default=3000, help="skip optional sparse direct solver above this size to limit fill-in risk")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--personalization", choices=["uniform", "random", "localized"], default="uniform")
    parser.add_argument("--dangling-distribution", choices=["personalization", "uniform", "random"], default="personalization")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args(argv)
    values = vars(args).copy()
    output = values.pop("output")
    for key in ("sizes", "families", "methods", "threads", "alphas"):
        values[key] = tuple(values[key])
    try:
        report = run_suite(Config(**values), progress=lambda message: print(message, flush=True))
    except ValueError as exc:
        parser.error(str(exc))
    files = write_report(report, output)
    for path in files:
        print(path)
    return 1 if any(item["status"] == "failed" for case in report["cases"] for item in case["methods"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
