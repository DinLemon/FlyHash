"""Benchmark datasets. Phase 1 needs MNIST only."""

from __future__ import annotations

import gzip
import io
import struct
import tarfile
import urllib.request
import zipfile
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


def _fetch_url(url: str, name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if not target.exists():
        temp_file = cache_dir / f"{name}.tmp"
        try:
            urllib.request.urlretrieve(url, temp_file)
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
    db = read_idx_images(_fetch_url(MNIST_BASE + TRAIN_IMAGES, TRAIN_IMAGES, cache))[:DATABASE_SIZE]
    if db.shape[0] != DATABASE_SIZE:
        raise ValueError(f"Expected {DATABASE_SIZE} images in {TRAIN_IMAGES}, got {db.shape[0]}")

    queries = read_idx_images(_fetch_url(MNIST_BASE + TEST_IMAGES, TEST_IMAGES, cache))[:QUERY_COUNT]
    if queries.shape[0] != QUERY_COUNT:
        raise ValueError(f"Expected {QUERY_COUNT} images in {TEST_IMAGES}, got {queries.shape[0]}")

    return db, queries


GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_NAME = "glove.6B.zip"
GLOVE_MEMBER = "glove.6B.50d.txt"
GLOVE_DIM = 50
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
SIFT_NAME = "siftsmall.tar.gz"
SIFT_QUERY_COUNT = 100  # siftsmall ships 100 queries, not 1000


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read a .fvecs file: each record is an int32 dimension then that many floats."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise ValueError(f"{path} has an inconsistent record dimension")
    records = raw.reshape(-1, dim + 1)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"{path} has an inconsistent record dimension")
    return records[:, 1:].copy().view(np.float32)


def load_glove(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]:
    """First 11000 GloVe word vectors: 10000 database, 1000 queries."""
    path = _fetch_url(GLOVE_URL, GLOVE_NAME, Path(cache_dir))
    needed = DATABASE_SIZE + QUERY_COUNT
    rows = []
    with zipfile.ZipFile(path) as archive, archive.open(GLOVE_MEMBER) as member:
        for line_num, raw in enumerate(io.TextIOWrapper(member, encoding="utf-8"), 1):
            parts = raw.rstrip().split(" ")
            values = [float(x) for x in parts[1:]]
            if len(values) != GLOVE_DIM:
                raise ValueError(f"{path} line {line_num}: expected {GLOVE_DIM} values, got {len(values)}")
            rows.append(values)
            if len(rows) == needed:
                break
    if len(rows) < needed:
        raise ValueError(f"{path} holds {len(rows)} vectors, need {needed}")
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:DATABASE_SIZE], arr[DATABASE_SIZE:needed]


def load_sift(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]:
    """SIFT-small: 10000 base vectors and its 100 shipped queries."""
    cache = Path(cache_dir)
    archive = _fetch_url(SIFT_URL, SIFT_NAME, cache)
    base = cache / "siftsmall" / "siftsmall_base.fvecs"
    if not base.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(cache, filter="data")
    db = read_fvecs(cache / "siftsmall" / "siftsmall_base.fvecs")[:DATABASE_SIZE]
    queries = read_fvecs(cache / "siftsmall" / "siftsmall_query.fvecs")[:SIFT_QUERY_COUNT]
    if db.shape[0] != DATABASE_SIZE:
        raise ValueError(f"SIFT base holds {db.shape[0]} vectors, need {DATABASE_SIZE}")
    if queries.shape[0] != SIFT_QUERY_COUNT:
        raise ValueError(f"{cache / 'siftsmall' / 'siftsmall_query.fvecs'} holds {queries.shape[0]} vectors, need {SIFT_QUERY_COUNT}")
    return db, queries


DATASETS = {"mnist": load_mnist, "glove": load_glove, "sift": load_sift}
