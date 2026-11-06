import concurrent.futures
import multiprocessing as mp
import os
import shutil
import tempfile

import numpy as np
import pytest
import tinytensor as tt


def _worker_compile_and_eval(cache_dir, seed):
    """Worker function executed in separate OS processes."""
    # Each process points to the shared cache_dir
    g = tt.Graph()
    x = g.input([4, 4], name="X")
    y = g.input([4, 4], name="Y")
    z = tt.relu(x + y)

    prog = tt.compile(g, outputs=[z], mode="scalar", cache_dir=cache_dir)
    data_x = np.full((4, 4), float(seed), dtype=np.float32)
    data_y = np.full((4, 4), 2.5, dtype=np.float32)
    res = prog.run({x: data_x, y: data_y}).numpy()

    expected = np.maximum(data_x + data_y, 0.0)
    assert np.allclose(res, expected, rtol=1e-5, atol=1e-5)
    return os.getpid(), float(np.mean(res)), True


@pytest.mark.parametrize("num_workers", [2, 4, 8, 16])
def test_multiprocess_kernel_cache_concurrency(num_workers):
    """Verify concurrent multi-process kernel compilation and publication safety across 2, 4, 8, 16 processes."""
    test_cache = tempfile.mkdtemp(prefix=f"tt_mp_cache_{num_workers}_")
    try:
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(_worker_compile_and_eval, test_cache, i) for i in range(num_workers)
            ]
            results = [f.result(timeout=60) for f in concurrent.futures.as_completed(futures)]

        assert len(results) == num_workers
        for pid, mean_val, status in results:
            assert status is True

        # Verify cache directory integrity: exactly one .so file, valid ELF header
        so_files = [f for f in os.listdir(test_cache) if f.endswith(".so")]
        assert len(so_files) == 1
        so_path = os.path.join(test_cache, so_files[0])
        assert os.path.getsize(so_path) >= 64
        with open(so_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"\x7fELF"

        # Verify no orphan temporary staging files left behind
        tmp_files = [f for f in os.listdir(test_cache) if ".tmp." in f]
        assert len(tmp_files) == 0

    finally:
        shutil.rmtree(test_cache, ignore_errors=True)


def _worker_distinct_kernels(cache_dir, rank):
    """Worker compiling distinct kernel graphs simultaneously."""
    g = tt.Graph()
    size = 4 + rank
    x = g.input([size, size], name="X")
    z = tt.relu(x)
    prog = tt.compile(g, outputs=[z], mode="scalar", cache_dir=cache_dir)
    data = np.full((size, size), float(rank) - 2.0, dtype=np.float32)
    out = prog.run({x: data}).numpy()
    expected = np.maximum(data, 0.0)
    assert np.allclose(out, expected)
    return rank, True


def test_multiprocess_distinct_kernels_parallel():
    """Verify multiple processes simultaneously compiling distinct kernels into shared cache."""
    test_cache = tempfile.mkdtemp(prefix="tt_mp_distinct_")
    try:
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=ctx) as executor:
            futures = [executor.submit(_worker_distinct_kernels, test_cache, i) for i in range(8)]
            results = [f.result(timeout=60) for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 8
        so_files = [f for f in os.listdir(test_cache) if f.endswith(".so")]
        assert len(so_files) == 8
    finally:
        shutil.rmtree(test_cache, ignore_errors=True)
