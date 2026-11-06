import numpy as np
import tinytensor as tt


def test_optimizer_constant_folding():
    g = tt.Graph()
    c1 = g.constant([2.0, 3.0])
    c2 = g.constant([4.0, 5.0])
    c3 = c1 * c2
    x = g.input([2])
    out = x + c3

    prog = tt.compile(g, outputs=[out], optimize=True)
    report = prog.optimization_report

    assert report.constant_foldings >= 1
    res = prog.run({x: [1.0, 1.0]})
    np.testing.assert_allclose(res.numpy(), [9.0, 16.0])


def test_optimizer_cse():
    g = tt.Graph()
    a = g.input([4])
    b = g.input([4])
    s1 = a + b
    s2 = a + b  # duplicate
    out = s1 * s2

    prog = tt.compile(g, outputs=[out], optimize=True)
    report = prog.optimization_report

    assert report.cse_eliminated >= 1
    res = prog.run({a: [1.0, 2.0, 3.0, 4.0], b: [1.0, 1.0, 1.0, 1.0]})
    np.testing.assert_allclose(res.numpy(), [4.0, 9.0, 16.0, 25.0])


def test_optimizer_fusion():
    g = tt.Graph()
    a = g.input([8])
    b = g.input([8])
    out = tt.relu(a + b)

    prog = tt.compile(g, outputs=[out], optimize=True)
    report = prog.optimization_report

    assert report.fused_groups >= 1
    res = prog.run({a: np.full(8, -5.0), b: np.full(8, 10.0)})
    np.testing.assert_allclose(res.numpy(), np.full(8, 5.0))


def test_optimizer_dead_code_elimination():
    g = tt.Graph()
    x = g.input([4])
    dead = x * x * x  # dead computation
    assert dead is not None
    out = tt.relu(x)

    prog = tt.compile(g, outputs=[out], optimize=True)
    report = prog.optimization_report

    assert report.dce_eliminated >= 1


def test_optimizer_memory_savings():
    g = tt.Graph()
    x = g.input([1000])
    t1 = x + 1.0
    t2 = tt.relu(t1)
    t3 = t2 * 2.0
    out = tt.sigmoid(t3)

    prog = tt.compile(g, outputs=[out], optimize=True)
    report = prog.optimization_report

    assert report.reused_buffers >= 1
    assert report.memory_savings_bytes > 0
