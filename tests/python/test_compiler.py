import numpy as np
import pytest
import tinytensor as tt


def test_compile_modes_agreement():
    g = tt.Graph()
    a = g.input([4, 4])
    b = g.input([4, 4])
    c = tt.relu(a @ b)

    a_np = np.random.randn(4, 4).astype(np.float32)
    b_np = np.random.randn(4, 4).astype(np.float32)

    prog_interp = tt.compile(g, outputs=[c], mode="interpreted")
    prog_scalar = tt.compile(g, outputs=[c], mode="scalar")
    prog_avx2 = tt.compile(g, outputs=[c], mode="avx2")

    res_interp = prog_interp.run({a: a_np, b: b_np})
    res_scalar = prog_scalar.run({a: a_np, b: b_np})
    res_avx2 = prog_avx2.run({a: a_np, b: b_np})

    np.testing.assert_allclose(res_interp.numpy(), res_scalar.numpy(), atol=1e-5)
    np.testing.assert_allclose(res_scalar.numpy(), res_avx2.numpy(), atol=1e-5)


def test_compiler_ir_inspection():
    g = tt.Graph()
    x = g.input([10, 20], name="X")
    y = tt.relu(x)
    prog = tt.compile(g, outputs=[y])

    ir_text = prog.ir
    assert "input<float32>[10, 20]" in ir_text
    assert "relu" in ir_text
    assert "return" in ir_text


def test_compiler_dot_graph():
    g = tt.Graph()
    x = g.input([2, 2])
    y = tt.relu(x)
    prog = tt.compile(g, outputs=[y])

    dot_text = prog.dot
    assert "digraph TinyTensorGraph" in dot_text


def test_compiler_repeated_run():
    g = tt.Graph()
    x = g.input([2])
    y = x + 1.0
    prog = tt.compile(g, outputs=[y])

    res1 = prog.run({x: [1.0, 2.0]})
    res2 = prog.run({x: [10.0, 20.0]})

    np.testing.assert_allclose(res1.numpy(), [2.0, 3.0])
    np.testing.assert_allclose(res2.numpy(), [11.0, 21.0])


def test_compiler_missing_input_raises():
    g = tt.Graph()
    x = g.input([2])
    y = g.input([2])
    z = x + y
    prog = tt.compile(g, outputs=[z])

    with pytest.raises(tt.RuntimeExecutionError):
        prog.run({x: [1.0, 2.0]})


def test_compiler_multi_output():
    g = tt.Graph()
    a = g.input([3])
    b = g.input([3])
    s = a + b
    p = a * b
    prog = tt.compile(g, outputs=[s, p])
    res_s, res_p = prog.run({a: [2.0, 3.0, 4.0], b: [10.0, 20.0, 30.0]})
    np.testing.assert_allclose(res_s.numpy(), [12.0, 23.0, 34.0])
    np.testing.assert_allclose(res_p.numpy(), [20.0, 60.0, 120.0])


def test_compiler_profiler_integration():
    g = tt.Graph()
    x = g.input([100, 100])
    y = g.input([100, 100])
    z = tt.relu(x @ y)
    prog = tt.compile(g, outputs=[z], mode="interpreted")

    prof = tt.Profiler()
    x_val = np.random.randn(100, 100).astype(np.float32)
    y_val = np.random.randn(100, 100).astype(np.float32)
    res = prog.run({x: x_val, y: y_val}, profiler=prof)
    assert res.shape == [100, 100]

    report = prof.summary()
    assert "TinyTensor Execution Profile" in report
    assert prof.total_time_us > 0


def test_compiler_scalar_mode_correctness():
    g = tt.Graph()
    x = g.input([16])
    y = g.input([16])
    out = (x + y) * (x - y)
    prog = tt.compile(g, outputs=[out], mode="scalar")
    x_np = np.arange(16, dtype=np.float32)
    y_np = np.ones(16, dtype=np.float32)
    res = prog.run({x: x_np, y: y_np})
    np.testing.assert_allclose(res.numpy(), (x_np + y_np) * (x_np - y_np))


def test_compiler_large_matmul():
    dim = 64
    g = tt.Graph()
    a = g.input([dim, dim])
    b = g.input([dim, dim])
    c = a @ b
    prog = tt.compile(g, outputs=[c])
    a_np = np.random.randn(dim, dim).astype(np.float32)
    b_np = np.random.randn(dim, dim).astype(np.float32)
    res = prog.run({a: a_np, b: b_np})
    np.testing.assert_allclose(res.numpy(), a_np @ b_np, atol=1e-3)


def test_compiler_diagnostic_apis():
    g = tt.Graph()
    x = g.input([4, 4], name="X")
    y = g.input([4, 4], name="Y")
    z = tt.relu(x + y)
    prog = tt.compile(g, outputs=[z], mode="avx2", optimize=True)

    orig_ir = prog.original_ir()
    assert "input<float32>[4, 4]" in orig_ir
    assert "add" in orig_ir

    opt_ir = prog.optimized_ir()
    assert len(opt_ir) > 0

    source = prog.generated_source()
    assert len(source) > 0
    assert "tinytensor_kernel_entry" in source
    assert "node_" in source

    plan = prog.execution_plan()
    assert "TinyTensor Execution Plan" in plan
    assert "Execution Mode: Compiled AVX2" in plan
    assert "Memory Planning:" in plan


def test_compiler_six_way_cross_backend_differential():
    import torch

    np.random.seed(123)
    dim = 8
    g = tt.Graph()
    a = g.input([dim, dim], name="A")
    b = g.input([dim, dim], name="B")
    c = tt.relu(a @ b + 2.0)

    a_np = np.random.randn(dim, dim).astype(np.float32)
    b_np = np.random.randn(dim, dim).astype(np.float32)
    feed = {a: a_np, b: b_np}

    # 1. NumPy reference
    ref_np = np.maximum(a_np @ b_np + 2.0, 0.0)

    # 2. PyTorch reference
    a_pt = torch.from_numpy(a_np)
    b_pt = torch.from_numpy(b_np)
    ref_pt = torch.relu(a_pt @ b_pt + 2.0).numpy()

    # 3. Interp unoptimized
    prog_interp_unopt = tt.compile(g, outputs=[c], mode="interpreted", optimize=False)
    res_interp_unopt = prog_interp_unopt.run(feed).numpy()

    # 4. Interp optimized
    prog_interp_opt = tt.compile(g, outputs=[c], mode="interpreted", optimize=True)
    res_interp_opt = prog_interp_opt.run(feed).numpy()

    # 5. Compiled Scalar
    prog_scalar = tt.compile(g, outputs=[c], mode="scalar", optimize=True)
    res_scalar = prog_scalar.run(feed).numpy()

    # 6. Compiled AVX2
    prog_avx2 = tt.compile(g, outputs=[c], mode="avx2", optimize=True)
    res_avx2 = prog_avx2.run(feed).numpy()

    # Numerical parity across all 6 implementations
    np.testing.assert_allclose(ref_np, ref_pt, atol=1e-5)
    np.testing.assert_allclose(res_interp_unopt, ref_np, atol=1e-5)
    np.testing.assert_allclose(res_interp_opt, ref_np, atol=1e-5)
    np.testing.assert_allclose(res_scalar, ref_np, atol=1e-5)
    np.testing.assert_allclose(res_avx2, ref_np, atol=1e-5)


def test_compiler_batched_matmul_compiled():
    import torch

    np.random.seed(42)
    batch, M, K, N = 2, 4, 8, 3
    g = tt.Graph()
    a = g.input([batch, M, K], name="A")
    b = g.input([batch, K, N], name="B")
    out = a @ b

    a_np = np.random.randn(batch, M, K).astype(np.float32)
    b_np = np.random.randn(batch, K, N).astype(np.float32)

    prog = tt.compile(g, outputs=[out], mode="avx2", optimize=True)
    res_tt = prog.run({a: a_np, b: b_np}).numpy()

    ref_np = a_np @ b_np
    ref_pt = (torch.from_numpy(a_np) @ torch.from_numpy(b_np)).numpy()

    np.testing.assert_allclose(res_tt, ref_np, atol=1e-4)
    np.testing.assert_allclose(res_tt, ref_pt, atol=1e-4)
