import gzip
import struct

import numpy as np
import pytest

from flyhash.datasets import load_mnist, read_idx_images


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
