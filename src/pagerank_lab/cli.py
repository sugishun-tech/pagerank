"""Command-line access to graph preparation, PageRank, and inspection."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

from . import generators
from .io import load_graph, load_vector, result_metadata, save_graph, save_result


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("graph", type=Path, help="edge list, sparse NPZ, or Matrix Market file")
    parser.add_argument("--format", choices=["auto", "edges", "npz", "mtx"], default="auto")
    parser.add_argument("--nodes", "--n", dest="n", type=int, help="number of nodes, including isolates")
    parser.add_argument("--delimiter", help="single-character edge-list delimiter")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pagerank-lab", description=__doc__)
    try:
        package_version = version("pagerank-lab")
    except PackageNotFoundError:
        package_version = "0.1.0 (source tree)"
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_version}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="compute PageRank and report an L1 error estimate")
    _input_arguments(run)
    run.add_argument("--alpha", type=float, default=0.85)
    run.add_argument("--tol", type=float, default=1e-10, help="target absolute L1 error; consider 1e-6 for float32")
    run.add_argument("--max-iter", type=int, default=1000)
    run.add_argument("--method", default="cython", help="solver method supported by pagerank()")
    run.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--partition", choices=["balanced", "rows"], default="balanced",
                     help="OpenMP work partition: balance edges and rows, or equal row counts")
    run.add_argument("--personalization", type=Path, help="teleport vector: JSON, NPY, NPZ, or text")
    run.add_argument("--dangling", type=Path, help="dangling redistribution vector; default is personalization")
    run.add_argument("--x0", type=Path, help="initial probability vector")
    run.add_argument("--history", action="store_true", help="retain convergence history")
    run.add_argument("--output", "-o", type=Path, help="result JSON, NPZ, or CSV (+ .meta.json)")
    run.add_argument("--top", type=int, default=10, help="number of leading nodes to display")

    inspect = commands.add_parser("inspect", help="inspect size, sparsity, dangling nodes, and components")
    _input_arguments(inspect)
    inspect.add_argument("--components", action="store_true", help="also compute strong/weak components")

    generate = commands.add_parser("generate", help="make a seeded synthetic graph")
    generate.add_argument("output", type=Path, help="NPZ, MTX, CSV, or text destination")
    generate.add_argument("--kind", choices=["uniform", "skewed", "block"], default="uniform")
    generate.add_argument("--nodes", "--n", dest="n", type=int, required=True)
    generate.add_argument("--edges", "--m", dest="m", type=int, required=True,
                          help="sampled edges before duplicate merging")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--dangling-fraction", type=float, default=0.05)
    generate.add_argument("--weighted", action="store_true")
    generate.add_argument("--exponent", type=float, default=1.2, help="skewed graph target popularity exponent")
    generate.add_argument("--blocks", type=int, default=4)
    generate.add_argument("--within-probability", type=float, default=0.9)
    return parser


def _inspect(graph, *, components: bool = False) -> dict:
    adjacency = graph.adjacency
    indegree = np.bincount(adjacency.indices, minlength=graph.n)
    outdegree = np.diff(adjacency.indptr)
    report = {
        "nodes": graph.n,
        "edges": graph.m,
        "density": float(graph.m / (graph.n * graph.n)),
        "dangling_nodes": int(np.count_nonzero(graph.dangling)),
        "isolated_nodes": int(np.count_nonzero((indegree == 0) & (outdegree == 0))),
        "max_in_degree": int(indegree.max(initial=0)),
        "max_out_degree": int(outdegree.max(initial=0)),
        "adjacency_bytes": int(adjacency.data.nbytes + adjacency.indices.nbytes + adjacency.indptr.nbytes),
    }
    if components:
        report["strong_components"] = int(connected_components(adjacency, directed=True, connection="strong", return_labels=False))
        report["weak_components"] = int(connected_components(adjacency, directed=True, connection="weak", return_labels=False))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            common = dict(seed=args.seed, dangling_fraction=args.dangling_fraction, weighted=args.weighted)
            if args.kind == "uniform":
                graph = generators.uniform_graph(args.n, args.m, **common)
            elif args.kind == "skewed":
                graph = generators.skewed_graph(args.n, args.m, exponent=args.exponent, **common)
            else:
                graph = generators.block_graph(args.n, args.m, blocks=args.blocks,
                                               within_probability=args.within_probability, **common)
            save_graph(graph, args.output)
            print(json.dumps({"output": str(args.output), "model": args.kind, "sampled_edges": args.m,
                              "seed": args.seed, **_inspect(graph)}, indent=2))
            return 0
        graph = load_graph(args.graph, format=args.format, n=args.n, delimiter=args.delimiter)
        if args.command == "inspect":
            print(json.dumps(_inspect(graph, components=args.components), indent=2))
            return 0
        if args.top < 0:
            raise ValueError("--top must be nonnegative")
        # Import here: graph conversion and CLI help do not require a solver run.
        from . import pagerank
        vectors = {
            name: load_vector(path, n=graph.n) if path is not None else None
            for name, path in (("personalization", args.personalization), ("dangling", args.dangling), ("x0", args.x0))
        }
        result = pagerank(graph, alpha=args.alpha, tol=args.tol, max_iter=args.max_iter,
                          method=args.method, dtype=args.dtype, threads=args.threads, partition=args.partition,
                          history=args.history, raise_on_failure=False, **vectors)
        if args.output:
            save_result(result, args.output)
        order = np.argsort(-result.scores, kind="stable")[:args.top]
        summary = {
            **result_metadata(result),
            "nodes": graph.n,
            "edges": graph.m,
            "top": [{"node": int(i), "score": float(result.scores[i])} for i in order],
        }
        if args.output:
            summary["output"] = str(args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.converged else 2
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"pagerank-lab: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

