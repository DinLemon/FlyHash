"""Benchmark datasets. Phase 1 needs MNIST only."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"

DATABASE_SIZE = 10_000
QUERY_COUNT = 1_000
IDX_IMAGE_MAGIC = 2051


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read a gzipped IDX3 image file into a (n, pixels) float32 array."""
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != IDX_IMAGE_MAGIC:
            raise ValueError(f"bad magic number {magic}, expected {IDX_IMAGE_MAGIC}")
        buf = f.read(count * rows * cols)
    flat = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return flat.reshape(count, rows * cols)


def _fetch(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if not target.exists():
        temp_file = cache_dir / f"{name}.tmp"
        try:
            urllib.request.urlretrieve(MNIST_BASE + name, temp_file)
            temp_file.replace(target)
        except Exception:
            temp_file.unlink(missing_ok=True)
            raise
    return target


def load_mnist(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]:
    """Return (database, queries).

    The database is the first 10 000 training images and the queries are the
    first 1000 test images, so a query is never its own nearest neighbour.
    """
    cache = Path(cache_dir)
    db = read_idx_images(_fetch(TRAIN_IMAGES, cache))[:DATABASE_SIZE]
    if db.shape[0] != DATABASE_SIZE:
        raise ValueError(f"Expected {DATABASE_SIZE} images in {TRAIN_IMAGES}, got {db.shape[0]}")

    queries = read_idx_images(_fetch(TEST_IMAGES, cache))[:QUERY_COUNT]
    if queries.shape[0] != QUERY_COUNT:
        raise ValueError(f"Expected {QUERY_COUNT} images in {TEST_IMAGES}, got {queries.shape[0]}")

    return db, queries
