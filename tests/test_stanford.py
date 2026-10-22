"""Tiny local fixtures for the external-dataset loader; no network or benchmark."""

import gzip
import hashlib

import numpy as np
import pytest

from benchmarks.stanford import download_stanford, load_stanford


def test_sorted_mapping_preserves_loops_and_aggregates_duplicates(tmp_path):
    path = tmp_path / "tiny.txt.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write("# tiny graph\n40 10\n10 10\n40 10\n70 40\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    adjacency, meta = load_stanford(path, expected_shape=None, expected_sha256=digest)
    np.testing.assert_array_equal(adjacency.toarray(), [[1, 0, 0], [2, 0, 0], [0, 1, 0]])
    assert meta["node_mapping"]["original_min"] == 10
    assert meta["node_mapping"]["original_max"] == 70
    assert meta["n"] == 3 and meta["m"] == 3 and meta["raw_edges"] == 4
    assert meta["self_loop_count"] == 1
    assert meta["source"]["gzip_sha256"] == digest
    assert not meta["source"]["official_dimensions_verified"]
    with pytest.raises(ValueError, match="dimensions"):
        load_stanford(path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_stanford(path, expected_shape=None, expected_sha256="0"*64)
    with pytest.raises(ValueError, match="canonical edge count"):
        load_stanford(path, expected_shape=(3, 4))


def test_existing_download_is_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "existing.gz"
    original = b"an unrelated existing file"
    path.write_bytes(original)

    def unexpected_network(*args, **kwargs):
        raise AssertionError("must not contact the network for an existing path")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_network)
    assert download_stanford(path) is False
    assert path.read_bytes() == original
