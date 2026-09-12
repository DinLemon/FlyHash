import gzip
import struct

import numpy as np
import pytest

from flyhash.datasets import (
    load_mnist,
    read_idx_images,
    TRAIN_IMAGES,
    TEST_IMAGES,
    DATABASE_SIZE,
    QUERY_COUNT,
    IDX_IMAGE_MAGIC,
)


def test_read_idx_images_parses_the_header_and_pixels(tmp_path):
    images = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    path = tmp_path / "toy-idx3-ubyte.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, 2, 4, 3))
        f.write(images.tobytes())
    out = read_idx_images(path)
    assert out.shape == (2, 12)
    np.testing.assert_array_equal(out, images.reshape(2, 12).astype(np.float32))


def test_read_idx_images_rejects_a_bad_magic_number(tmp_path):
    path = tmp_path / "bad-idx3-ubyte.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 1234, 1, 1, 1))
        f.write(b"\x00")
    with pytest.raises(ValueError, match="magic"):
        read_idx_images(path)


@pytest.mark.network
def test_load_mnist_returns_the_preregistered_shapes(tmp_path):
    db, queries = load_mnist(cache_dir=tmp_path)
    assert db.shape == (10000, 784)
    assert queries.shape == (1000, 784)
    assert db.dtype == np.float32
    assert db.max() <= 255.0


def test_load_mnist_rejects_a_short_file(tmp_path):
    # Create a short training file with fewer images than DATABASE_SIZE
    short_count = DATABASE_SIZE // 2
    images = np.zeros((short_count, 28, 28), dtype=np.uint8)
    train_path = tmp_path / TRAIN_IMAGES
    with gzip.open(train_path, "wb") as f:
        f.write(struct.pack(">IIII", IDX_IMAGE_MAGIC, short_count, 28, 28))
        f.write(images.tobytes())

    # Create a valid test file to avoid network call
    test_images = np.zeros((QUERY_COUNT, 28, 28), dtype=np.uint8)
    test_path = tmp_path / TEST_IMAGES
    with gzip.open(test_path, "wb") as f:
        f.write(struct.pack(">IIII", IDX_IMAGE_MAGIC, QUERY_COUNT, 28, 28))
        f.write(test_images.tobytes())

    with pytest.raises(ValueError, match="Expected"):
        load_mnist(cache_dir=tmp_path)


import struct as _struct

from flyhash.datasets import DATASETS, read_fvecs


def test_read_fvecs_parses_dimension_prefixed_records(tmp_path):
    path = tmp_path / "toy.fvecs"
    with open(path, "wb") as f:
        for row in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]):
            f.write(_struct.pack("<i", 3))
            f.write(_struct.pack("<3f", *row))
    out = read_fvecs(path)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


def test_read_fvecs_rejects_inconsistent_dimensions(tmp_path):
    path = tmp_path / "bad.fvecs"
    with open(path, "wb") as f:
        f.write(_struct.pack("<i", 3))
        f.write(_struct.pack("<3f", 1.0, 2.0, 3.0))
        f.write(_struct.pack("<i", 2))
        f.write(_struct.pack("<2f", 4.0, 5.0))
    with pytest.raises(ValueError, match="dimension"):
        read_fvecs(path)


def test_dataset_registry_lists_all_three():
    assert sorted(DATASETS) == ["glove", "mnist", "sift"]


# These two use the project's real cache directory rather than tmp_path on
# purpose: the GloVe archive is 862 MB and must not be re-downloaded per test.
@pytest.mark.network
def test_load_glove_shapes():
    db, q = DATASETS["glove"]()
    assert db.shape == (10000, 50)
    assert q.shape == (1000, 50)


@pytest.mark.network
def test_load_sift_shapes():
    db, q = DATASETS["sift"]()
    assert db.shape == (10000, 128)
    assert q.shape == (100, 128)
