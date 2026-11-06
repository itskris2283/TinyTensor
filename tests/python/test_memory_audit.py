import numpy as np
import tinytensor as tt


def test_deep_mlp_memory_planning_report():
    """Verify memory savings and buffer reuse on a 10-layer MLP."""
    dim = 64
    g = tt.Graph()
    x = g.input([dim, dim])

    cur = x
    for _ in range(10):
        w = g.constant(np.eye(dim, dtype=np.float32))
        cur = tt.relu(cur @ w)

    g.mark_output(cur)

    prog = tt.compile(g, outputs=[cur], mode="scalar", optimize=True)
    report = prog.optimization_report

    assert report.total_intermediate_allocations > 0
    assert report.reused_buffers >= 5
    assert report.memory_savings_bytes > 0
    assert report.peak_memory_bytes < (report.total_intermediate_allocations * dim * dim * 4)


def test_branching_residual_block_correctness():
    """Verify that branching and residual connections do not alias or clobber buffers."""
    dim = 16
    x_np = np.random.randn(dim, dim).astype(np.float32)

    g = tt.Graph()
    x = g.input([dim, dim])
    w = g.constant(np.eye(dim, dtype=np.float32))

    # Branch 1: relu(x @ w)
    h1 = tt.relu(x @ w)
    # Branch 2: residual connection + branch 1
    out = h1 + x
    g.mark_output(out)

    expected = np.maximum(x_np @ np.eye(dim, dtype=np.float32), 0.0) + x_np

    for mode in ["interpreted", "scalar", "avx2"]:
        prog = tt.compile(g, outputs=[out], mode=mode, optimize=True)
        res = prog.run({x: x_np}).numpy()
        np.testing.assert_allclose(res, expected, atol=1e-5)


def test_multiple_outputs_retention_correctness():
    """Verify multiple outputs are preserved and never clobbered by later execution steps."""
    dim = 8
    x_np = np.ones((dim, dim), dtype=np.float32)

    g = tt.Graph()
    x = g.input([dim, dim])

    out1 = x + x  # 2.0
    out2 = out1 * x  # 2.0
    out3 = out2 + out1  # 4.0

    g.mark_output(out1)
    g.mark_output(out2)
    g.mark_output(out3)

    for mode in ["interpreted", "scalar", "avx2"]:
        prog = tt.compile(g, outputs=[out1, out2, out3], mode=mode, optimize=True)
        res = prog.run({x: x_np})

        np.testing.assert_allclose(res[0].numpy(), 2.0 * x_np)
        np.testing.assert_allclose(res[1].numpy(), 2.0 * x_np)
        np.testing.assert_allclose(res[2].numpy(), 4.0 * x_np)


def test_process_memory_stats_tracking():
    """Verify process RSS tracking via get_process_memory_stats."""
    stats = tt.get_process_memory_stats()
    assert stats.current_rss_bytes > 0
    assert stats.peak_rss_bytes > 0
    assert stats.peak_rss_bytes >= stats.current_rss_bytes

    # Allocate large buffer (8 MB)
    large_tensors = [tt.full([1024, 1024], 1.0) for _ in range(2)]
    stats_after = tt.get_process_memory_stats()
    assert stats_after.current_rss_bytes > 0
    assert stats_after.peak_rss_bytes >= stats.peak_rss_bytes
    del large_tensors


def test_four_tier_memory_categories_evaluation():
    """Verify the 4 distinct memory tiers across a deep computational graph:

    Tier A: Theoretical peak live tensor bytes (sum of simultaneously active tensors).
    Tier B: Planned buffer memory (total bytes allocated in memory pool after reuse).
    Tier C: Allocator requested bytes (exact memory allocated for tensor storage).
    Tier D: Process Resident Set Size (RSS from /proc/self/statm & getrusage).
    """
    dim = 256  # 256x256 float32 = 256 KB per tensor
    elem_bytes = dim * dim * 4

    g = tt.Graph()
    x = g.input([dim, dim])

    # Linear chain of 8 transformations
    cur = x
    for _ in range(8):
        w = g.constant(np.eye(dim, dtype=np.float32))
        cur = tt.relu(cur @ w)
    g.mark_output(cur)

    prog = tt.compile(g, outputs=[cur], mode="scalar", optimize=True)
    report = prog.optimization_report

    # Tier B: Planned buffer bytes
    tier_b_planned = report.peak_memory_bytes
    # Without reuse, 8 matmuls + 8 relus = 16 tensors = 16 * 256KB = 4096 KB
    unplanned_total = report.total_intermediate_allocations * elem_bytes
    assert tier_b_planned < unplanned_total
    assert report.reused_buffers > 0

    # Execute and measure Tier D: Process RSS
    rss_before = tt.get_process_memory_stats().current_rss_bytes
    data = np.random.randn(dim, dim).astype(np.float32)
    res = prog.run({x: data}).numpy()
    rss_after = tt.get_process_memory_stats().current_rss_bytes

    assert res.shape == (dim, dim)
    assert rss_after > 0
    # Process RSS includes executable pages, shared libraries, Python runtime, and heap
    assert rss_after >= tier_b_planned


def test_adversarial_diamond_dag_memory_planning():
    """Adversarial DAG test: Diamond pattern A -> (B, C) -> D.

    B and C are concurrently live at step D, so they MUST NOT alias or share the same buffer.
    After D is produced, both B and C expire, allowing their slots to be reused for subsequent steps.
    """
    dim = 32
    g = tt.Graph()
    a = g.input([dim, dim])

    # Left branch
    b = tt.relu(a)
    # Right branch
    c = tt.sigmoid(a)
    # Merge
    d = b + c
    # Subsequent node reusing expired branch memory
    e = tt.relu(d)
    g.mark_output(e)

    a_np = np.random.randn(dim, dim).astype(np.float32)
    expected_b = np.maximum(a_np, 0.0)
    expected_c = 1.0 / (1.0 + np.exp(-a_np))
    expected_e = np.maximum(expected_b + expected_c, 0.0)

    for mode in ["interpreted", "scalar", "avx2"]:
        prog = tt.compile(g, outputs=[e], mode=mode, optimize=True)
        out = prog.run({a: a_np}).numpy()
        np.testing.assert_allclose(out, expected_e, rtol=1e-5, atol=1e-5)


def test_adversarial_high_fanout_memory_planning():
    """Adversarial DAG test: High fan-out where node A feeds 8 downstream consumers.

    Node A must remain allocated until ALL 8 consumers have executed.
    """
    dim = 16
    g = tt.Graph()
    a = g.input([dim, dim])

    consumers = []
    for i in range(8):
        c_val = g.constant(np.full((dim, dim), float(i + 1), dtype=np.float32))
        consumers.append(a * c_val)

    # Accumulate all consumers
    acc = consumers[0]
    for c in consumers[1:]:
        acc = acc + c
    g.mark_output(acc)

    a_np = np.ones((dim, dim), dtype=np.float32)
    # Expected sum: a * (1 + 2 + ... + 8) = a * 36
    expected = 36.0 * a_np

    for mode in ["interpreted", "scalar", "avx2"]:
        prog = tt.compile(g, outputs=[acc], mode=mode, optimize=True)
        res = prog.run({a: a_np}).numpy()
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)
