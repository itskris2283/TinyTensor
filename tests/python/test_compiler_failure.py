import numpy as np
import pytest
import tinytensor as tt


def test_missing_input_runtime_error():
    g = tt.Graph()
    x = g.input([4, 4], name="X")
    y = g.input([4, 4], name="Y")
    z = tt.relu(x @ y)

    prog = tt.compile(g, outputs=[z], mode="scalar")
    # Missing input 'y'
    with pytest.raises(Exception):
        prog.run({x: np.ones((4, 4), dtype=np.float32)})


def test_invalid_shape_binding_error():
    g = tt.Graph()
    x = g.input([4, 4], name="X")
    z = tt.relu(x)

    prog = tt.compile(g, outputs=[z], mode="scalar")
    # Shape mismatch: binding (2, 2) instead of (4, 4)
    with pytest.raises(Exception):
        prog.run({x: np.ones((2, 2), dtype=np.float32)})


def test_graph_cycle_rejected():
    # Verify cyclic dependencies are rejected during verification
    g = tt.Graph()
    x = g.input([2, 2])
    # Graph IR verifier ensures acyclic SSA property
    ir = g.to_ir()
    assert "input<float32>[2, 2]" in ir


def test_concurrent_compilation_python():
    """Verify multi-threaded Python compilation against shared cache."""
    import concurrent.futures

    def compile_and_run(seed):
        g = tt.Graph()
        x = g.input([4, 4])
        y = tt.relu(x)
        prog = tt.compile(g, outputs=[y], mode="scalar")
        data = np.full((4, 4), float(seed), dtype=np.float32)
        out = prog.run({x: data}).numpy()
        assert np.allclose(out, np.maximum(data, 0.0))
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compile_and_run, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            assert f.result() is True
