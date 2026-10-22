"""Safe on-disk formats with explicit node identities and dangling vertices.

Rows denote sources and columns denote targets. Numeric IDs are 0..n-1.
NPZ and Matrix Market preserve the complete matrix shape, including isolates.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import io as scipy_io, sparse

from .graph import Graph


def _format(path: Path, format: str) -> str:
    if format != "auto":
        if format not in {"npz", "mtx", "edges"}:
            raise ValueError("format must be 'auto', 'npz', 'mtx', or 'edges'")
        return format
    return {".npz": "npz", ".mtx": "mtx"}.get(path.suffix.lower(), "edges")


def load_graph(
    path: str | Path,
    *,
    format: str = "auto",
    n: int | None = None,
    delimiter: str | None = None,
) -> Graph:
    """Load CSR NPZ, Matrix Market, or a numeric edge list.

    Text permits comments starting with ``#``, an optional
    ``source,target[,weight]`` header, and ``# n=12`` to retain isolates.
    CSV uses commas by default; other text uses whitespace. Repeated edges
    are summed by Graph. For a headerless empty graph, supply n explicitly.
    """
    path = Path(path)
    format = _format(path, format)
    if format != "edges":
        matrix = sparse.load_npz(path) if format == "npz" else scipy_io.mmread(path)
        if n is not None and matrix.shape != (n, n):
            raise ValueError(f"stored matrix shape {matrix.shape} disagrees with n={n}")
        return Graph.from_scipy(matrix)
    if delimiter is None and path.suffix.lower() == ".csv":
        delimiter = ","
    if delimiter is not None and len(delimiter) != 1:
        raise ValueError("delimiter must be one character")
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    declared_n: int | None = None
    seen_data = False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = re.fullmatch(r"#\s*n\s*=\s*(\d+)\s*", line)
                if match:
                    value = int(match.group(1))
                    if declared_n is not None and declared_n != value:
                        raise ValueError(f"line {lineno}: inconsistent n metadata")
                    declared_n = value
                continue
            fields = next(csv.reader([line], delimiter=delimiter)) if delimiter else line.split()
            fields = [x.strip() for x in fields]
            if not seen_data and fields in (["source", "target"], ["source", "target", "weight"]):
                seen_data = True
                continue
            if len(fields) not in (2, 3):
                raise ValueError(f"line {lineno}: expected source,target[,weight]")
            try:
                if not re.fullmatch(r"[0-9]+", fields[0]) or not re.fullmatch(r"[0-9]+", fields[1]):
                    raise ValueError("node IDs must be nonnegative decimal integers")
                source, target = int(fields[0]), int(fields[1])
                weight = float(fields[2]) if len(fields) == 3 else 1.0
                if not np.isfinite(weight) or weight < 0:
                    raise ValueError("weight must be finite and nonnegative")
            except ValueError as exc:
                raise ValueError(f"line {lineno}: {exc}") from exc
            sources.append(source)
            targets.append(target)
            weights.append(weight)
            seen_data = True
    if declared_n is not None:
        if n is not None and n != declared_n:
            raise ValueError(f"file declares n={declared_n}, but n={n} was requested")
        n = declared_n
    return Graph.from_edges(sources, targets, weights=weights, n=n)


def save_graph(graph: Graph, path: str | Path, *, format: str = "auto") -> None:
    """Write a graph without densifying it; text stores an n metadata comment."""
    path = Path(path)
    format = _format(path, format)
    if format == "npz":
        # Passing an open file prevents NumPy silently appending '.npz'.
        with path.open("wb") as stream:
            sparse.save_npz(stream, graph.adjacency)
    elif format == "mtx":
        scipy_io.mmwrite(path, graph.adjacency, symmetry="general")
    else:
        matrix = graph.adjacency.tocoo()
        delimiter = "," if path.suffix.lower() == ".csv" else " "
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(f"# n={graph.n}\n")
            writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
            writer.writerow(["source", "target", "weight"])
            for source, target, weight in zip(matrix.row, matrix.col, matrix.data):
                writer.writerow([int(source), int(target), repr(float(weight))])


def load_vector(path: str | Path, *, n: int | None = None) -> np.ndarray:
    """Load a JSON array, NPY, NPZ ('scores' or 'vector'), or numeric text.

    NumPy pickle deserialization is always disabled. Values are left
    unnormalized so the solver can validate and normalize probability inputs.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        vector = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "scores" if "scores" in archive else "vector"
            if key not in archive:
                raise ValueError("vector NPZ must contain 'scores' or 'vector'")
            vector = archive[key]
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "scores" not in data:
                raise ValueError("JSON object must contain a scores array")
            data = data["scores"]
        vector = np.asarray(data)
    else:
        vector = np.loadtxt(path, ndmin=1)
    if np.iscomplexobj(vector):
        raise ValueError("vector must contain real values")
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if n is not None and vector.shape != (n,):
        raise ValueError(f"vector has length {len(vector)}, expected {n}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector contains nonfinite values")
    return vector


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def result_metadata(result: Any) -> dict[str, Any]:
    """Extract portable metadata while keeping the score vector separate."""
    names = (
        "iterations", "converged", "residual_l1", "error_bound_l1",
        "elapsed_seconds", "history", "method", "alpha", "tol", "dtype", "threads", "delta_l1", "partition",
    )
    return {name: _jsonable(getattr(result, name)) for name in names if hasattr(result, name)}


def save_result(result: Any, path: str | Path) -> None:
    """Export JSON, NPZ, or CSV plus a sibling .meta.json file.

    NPZ metadata is a Unicode JSON scalar and does not require pickle.
    CSV sorts by node ID, never by score, so a vector can be reconstructed.
    """
    path = Path(path)
    metadata = result_metadata(result)
    scores = np.asarray(result.scores)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps({"scores": scores.tolist(), **metadata}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif path.suffix.lower() == ".npz":
        with path.open("wb") as stream:
            np.savez_compressed(stream, scores=scores, metadata=json.dumps(metadata, allow_nan=False))
    elif path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["node", "score"])
            writer.writerows((i, repr(float(value))) for i, value in enumerate(scores))
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        raise ValueError("result output extension must be .json, .npz, or .csv")

