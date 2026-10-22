"""Benchmark the SNAP Stanford web graph collected in 2002.

Run from the checkout with an existing file::

    python -m benchmarks.stanford --input /path/to/web-Stanford.txt.gz

Network access occurs only when explicitly requested::

    python -m benchmarks.stanford --download /path/to/web-Stanford.txt.gz

The raw dataset is not included in this project.  Downloads never overwrite
an existing path.  Node IDs are remapped in sorted order; self-loops are kept.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

import numpy as np
from scipy import sparse

from .run import Config, benchmark_graph, environment, write_report


SOURCE_URL = "https://snap.stanford.edu/data/web-Stanford.html"
DOWNLOAD_URL = "https://snap.stanford.edu/data/web-Stanford.txt.gz"
EXPECTED_SHAPE = (281903, 2312497)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_stanford(destination: str | Path) -> bool:
    """Download only to an absent path; return False if a file already exists.

    A temporary file and atomic hard-link creation prevent clobbering a file
    created concurrently.  An existing file is validated by the loader later;
    it is never replaced, even if it has the wrong checksum or dimensions.
    """
    destination = Path(destination)
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"download destination is not a regular file: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=destination.name + ".", suffix=".part",
            dir=destination.parent, delete=False,
        ) as output:
            temporary_path = Path(output.name)
            request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "pagerank-lab/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        # Verify gzip integrity before making a completed download visible.
        with gzip.open(temporary_path, "rb") as check:
            for _ in iter(lambda: check.read(1024 * 1024), b""):
                pass
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"destination appeared during download; refusing to overwrite {destination}") from exc
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_stanford(
    path: str | Path, *, seed: int = 20261023,
    expected_shape: tuple[int, int] | None = EXPECTED_SHAPE,
    expected_sha256: str | None = None,
):
    """Return canonical CSR and provenance metadata from a gzipped edge list.

    ``expected_shape`` means (number of observed vertices, number of edges).
    The command-line workflow always enforces the official dimensions; None
    exists only to exercise the loader with small fixtures.  Both raw edge
    count and canonical nonzero count are checked for the official file.
    """
    path = Path(path)
    compressed_sha256 = _sha256(path)
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        if compressed_sha256 != expected_sha256:
            raise ValueError(f"gzip SHA-256 mismatch: expected {expected_sha256}, got {compressed_sha256}")
    with gzip.open(path, "rt", encoding="ascii") as handle:
        edges = np.loadtxt(handle, dtype=np.int64, comments="#", ndmin=2)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        raise ValueError("edge list must contain nonempty source/target integer columns")
    if np.any(edges < 0):
        raise ValueError("node IDs must be nonnegative integers")
    original_ids, dense = np.unique(edges, return_inverse=True)
    # NumPy releases differ in the shape returned by return_inverse.
    dense = dense.reshape(edges.shape)
    n, raw_m = int(original_ids.size), int(edges.shape[0])
    if expected_shape is not None and (n, raw_m) != expected_shape:
        raise ValueError(
            f"dataset dimensions do not match SNAP web-Stanford: "
            f"observed (n, raw_edges)={(n, raw_m)}, expected {expected_shape}"
        )
    adjacency = sparse.csr_matrix(
        (np.ones(raw_m, dtype=np.float64), (dense[:, 0], dense[:, 1])),
        shape=(n, n), dtype=np.float64,
    )
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    adjacency.eliminate_zeros()
    if expected_shape is not None and adjacency.nnz != expected_shape[1]:
        raise ValueError(
            f"canonical edge count {adjacency.nnz} differs from the official "
            f"{expected_shape[1]} (possible duplicate input edges)"
        )
    digest = hashlib.sha256()
    for arr in (adjacency.indptr.astype("<i8"), adjacency.indices.astype("<i8"), adjacency.data.astype("<f8")):
        digest.update(arr.tobytes())
    metadata = {
        "family": "web_stanford", "n": n, "m": int(adjacency.nnz),
        "sampled_edges": raw_m, "raw_edges": raw_m, "seed": int(seed), "weighted": False,
        "self_loops": "retained", "self_loop_count": int(np.count_nonzero(adjacency.diagonal())),
        "duplicate_edges": "summed", "generator": None,
        "adjacency_sha256": digest.hexdigest(),
        "source": {
            "name": "SNAP Stanford web graph", "page_url": SOURCE_URL,
            "download_url": DOWNLOAD_URL, "collected_year": 2002,
            "directed": True, "original_file_name": path.name,
            "gzip_size_bytes": path.stat().st_size, "gzip_sha256": compressed_sha256,
            "expected_nodes": EXPECTED_SHAPE[0], "expected_edges": EXPECTED_SHAPE[1],
            "official_dimensions_verified": expected_shape == EXPECTED_SHAPE,
        },
        "node_mapping": {
            "rule": "sorted unique observed source and target IDs -> contiguous 0..n-1",
            "original_min": int(original_ids[0]), "original_max": int(original_ids[-1]),
            "observed_unique_ids": n,
            "sorted_original_ids_sha256": hashlib.sha256(original_ids.astype("<i8").tobytes()).hexdigest(),
            "mapping_stored": False,
            "isolates_policy": "only IDs appearing in the official edge list; official node count is checked",
        },
    }
    return adjacency, metadata


def run_stanford(path: str | Path, config: Config, *, expected_sha256=None, progress=None):
    """Load the official graph, then reuse the common benchmark harness."""
    config.validate()
    started = datetime.now(timezone.utc).isoformat()
    load_start = time.perf_counter()
    adjacency, metadata = load_stanford(path, seed=config.seed, expected_sha256=expected_sha256)
    load_seconds = time.perf_counter() - load_start
    # Synthetic graph-generator fields have no meaning for this experiment.
    settings = asdict(config)
    for field in ("sizes", "families", "degree", "weighted", "python_max_n", "direct_max_n"):
        settings.pop(field)
    settings["dataset"] = "web-Stanford"
    report = {
        "schema_version": 1, "experiment": "real_web_graph",
        "started_at_utc": started, "config": settings, "environment": environment(),
        "dataset_loading_seconds": load_seconds,
        "timing_notes": [
            "Official SNAP web-Stanford graph, collected in 2002; this is one observed web graph, not the modern web.",
            "Network download, gzip parsing, ID remapping and canonical CSR creation are outside all solve and preprocessing times; loading is separately reported.",
            "Original IDs are renumbered in sorted order. Self-loops are retained. Observed vertex count and raw/canonical edge counts must match the official dimensions.",
            "Gzip SHA-256 and canonical CSR SHA-256 identify the exact measured input; dimensional checks alone are not cryptographic source authentication.",
            "Primary wall times are full public solve calls on prepared graph objects, including output ndarray conversion; graph preparation is separately timed.",
            "NetworkX internally converts its prepared DiGraph into a sparse matrix during each pagerank call; that work is included in its API timing.",
            "Cython/SciPy solver-only time is a separate field and excludes their input-wrapper and final residual validation cost.",
            "All runs use float64 and the same independently evaluated L1 residual/(1-alpha) target, nonnegativity and mass checks.",
            "igraph PRPACK has its own internal stopping rule; results are accepted only if the common post-check passes.",
            "Method order is shuffled for each repetition using a recorded seed. Warmups are discarded. GC remains enabled.",
            "preprocess_plus_median_seconds is the sum of separate measurements, not an independently timed cold end-to-end run.",
            "Process peak RSS is cumulative for the entire process and cannot be attributed to one method.",
            "Floating-point residual estimates are not interval-arithmetic certificates.",
        ],
        "cases": [],
    }
    for alpha in config.alphas:
        if progress:
            progress(f"web_stanford: n={metadata['n']:,} m={metadata['m']:,} alpha={alpha}; loaded in {load_seconds:.3f} s")
        case = benchmark_graph(adjacency, metadata, config, alpha)
        report["cases"].append(case)
        if progress:
            for name, item in case["methods"].items():
                median = item.get("summary", {}).get("median_seconds")
                progress(f"  {name}: {item['status']}" +
                         (f" {median*1000:.3f} ms" if median is not None else f" {item.get('reason', '')}"))
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="existing web-Stanford.txt.gz; never downloads")
    source.add_argument("--download", type=Path, metavar="PATH", help="explicitly download to PATH if absent; never overwrite")
    parser.add_argument("--expect-sha256", help="require this gzip SHA-256 before parsing")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/stanford"))
    parser.add_argument("--methods", nargs="+", choices=["cython", "scipy", "networkx", "igraph"], default=["cython", "scipy", "networkx", "igraph"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--alpha", type=float, default=.85)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args(argv)
    config = Config(methods=tuple(args.methods), threads=tuple(args.threads),
                    alphas=(args.alpha,), repeats=args.repeats, warmups=args.warmups,
                    tol=args.tol, max_iter=args.max_iter, seed=args.seed)
    try:
        config.validate()
        if args.download is not None:
            downloaded = download_stanford(args.download)
            print(f"{'Downloaded' if downloaded else 'Reusing existing file'}: {args.download}", flush=True)
        path = args.input if args.input is not None else args.download
        report = run_stanford(path, config, expected_sha256=args.expect_sha256,
                              progress=lambda message: print(message, flush=True))
    except (OSError, EOFError, ValueError) as exc:
        parser.error(str(exc))
    for path in write_report(report, args.output):
        print(path)
    return int(any(item["status"] == "failed" for case in report["cases"] for item in case["methods"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
