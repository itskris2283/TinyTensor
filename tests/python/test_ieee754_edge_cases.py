import numpy as np
import pytest
import tinytensor as tt


def float32_ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the difference in Units in the Last Place (ULPs) between two float32 arrays."""
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    ai = a32.view(np.int32).astype(np.int64)
    bi = b32.view(np.int32).astype(np.int64)
    # Map sign-magnitude to lexicographical order across positive and negative
    ai_ord = np.where(ai < 0, 0x80000000 - ai, ai)
    bi_ord = np.where(bi < 0, 0x80000000 - bi, bi)
    return np.abs(ai_ord - bi_ord)


def compute_numerical_metrics(actual: np.ndarray, expected: np.ndarray):
    """Compute max absolute error, max relative error, RMSE, and max ULP distance."""
    act = np.asarray(actual, dtype=np.float32)
    exp = np.asarray(expected, dtype=np.float32)
    diff = np.abs(act - exp)
    max_abs_err = float(np.max(diff))
    denom = np.maximum(np.abs(exp), 1e-12)
    max_rel_err = float(np.max(diff / denom))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ulps = float32_ulp_distance(act, exp)
    max_ulp = int(np.max(ulps))
    return {
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "rmse": rmse,
        "max_ulp": max_ulp,
    }


def test_special_floats_eager():
    """Verify IEEE 754 special values in eager operations against NumPy."""
    subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    vals = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            subnormal,
            -subnormal,
            1e38,
            -1e38,
            1.0,
            -1.0,
        ],
        dtype=np.float32,
    )

    t = tt.tensor(vals)
    # Division by zero
    zero_t = tt.tensor(np.zeros_like(vals))
    div_res = (t / zero_t).numpy()
    np_div = np.divide(vals, np.zeros_like(vals))

    for actual, expected in zip(div_res, np_div):
        if np.isnan(expected):
            assert np.isnan(actual)
        elif np.isposinf(expected):
            assert np.isposinf(actual)
        elif np.isneginf(expected):
            assert np.isneginf(actual)


def test_relu_semantic_contract():
    """Verify ReLU semantic contract across all backends.

    Contract: (x > 0.0f) ? x : 0.0f.
    - NaN evaluates to False -> 0.0f
    - -0.0f evaluates to False -> +0.0f
    - -inf -> 0.0f
    - +inf -> +inf
    """
    subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    edge_vals = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
            0.0,
            -0.0,
            subnormal,
            -subnormal,
            1e38,
            -1e38,
            42.5,
            -42.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            3.0,
        ],
        dtype=np.float32,
    )

    # Graph & Backends
    g = tt.Graph()
    x = g.input([len(edge_vals)])
    y = tt.relu(x)

    modes = ["interpreted", "scalar"]
    if tt.has_avx2():
        modes.append("avx2")

    for mode in modes:
        prog = tt.compile(g, outputs=[y], mode=mode)
        out = prog.run({x: edge_vals}).numpy()

        for idx, (val, res) in enumerate(zip(edge_vals, out)):
            if np.isnan(val):
                assert res == 0.0, f"[{mode}] NaN must yield 0.0, got {res}"
            elif np.isposinf(val):
                assert np.isposinf(res), f"[{mode}] +inf must yield +inf, got {res}"
            elif val <= 0.0:
                assert res == 0.0, f"[{mode}] <=0 must yield 0.0, got {res}"
                # Must be positive zero
                assert np.signbit(res) == 0, f"[{mode}] must be positive zero"
            else:
                assert res == val, f"[{mode}] >0 must preserve value, got {res}"


def test_sigmoid_numerical_stability():
    """Verify numerically stable sigmoid across range and extremes."""
    in_vals = np.array([100.0, -100.0, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], dtype=np.float32)

    g = tt.Graph()
    x = g.input([len(in_vals)])
    y = tt.sigmoid(x)

    modes = ["interpreted", "scalar"]
    for mode in modes:
        prog = tt.compile(g, outputs=[y], mode=mode)
        out = prog.run({x: in_vals}).numpy()

        # +100 -> 1.0
        assert out[0] == 1.0, f"[{mode}] sigmoid(+100) must be 1.0"
        # -100 -> near 0.0, subnormal epsilon
        assert 0.0 <= out[1] < 1e-12, f"[{mode}] sigmoid(-100) must be ~0.0"
        # +inf -> 1.0
        assert out[2] == 1.0, f"[{mode}] sigmoid(+inf) must be 1.0"
        # -inf -> 0.0
        assert out[3] == 0.0, f"[{mode}] sigmoid(-inf) must be 0.0"
        # 0.0 -> 0.5
        assert np.isclose(out[4], 0.5), f"[{mode}] sigmoid(0.0) must be 0.5"
        # -0.0 -> 0.5
        assert np.isclose(out[5], 0.5), f"[{mode}] sigmoid(-0.0) must be 0.5"


def test_fma_vs_nofma_numerical_ulp():
    """Verify FMA vs non-FMA agreement and compute detailed error metrics."""
    if not tt.has_fma3():
        pytest.skip("FMA3 not supported on this host")

    np.random.seed(42)
    dim = 32
    a_np = np.random.uniform(0.1, 10.0, size=(dim, dim)).astype(np.float32)
    b_np = np.random.uniform(0.1, 10.0, size=(dim, dim)).astype(np.float32)

    g = tt.Graph()
    x = g.input([dim, dim])
    y = g.input([dim, dim])
    z = x @ y

    prog_fma = tt.compile(g, outputs=[z], mode="avx2_fma")
    prog_nofma = tt.compile(g, outputs=[z], mode="avx2_nofma")
    prog_scalar = tt.compile(g, outputs=[z], mode="scalar")

    res_fma = prog_fma.run({x: a_np, y: b_np}).numpy()
    res_nofma = prog_nofma.run({x: a_np, y: b_np}).numpy()
    res_scalar = prog_scalar.run({x: a_np, y: b_np}).numpy()

    metrics_fma_nofma = compute_numerical_metrics(res_fma, res_nofma)
    metrics_nofma_scalar = compute_numerical_metrics(res_nofma, res_scalar)

    # ULP difference between FMA and non-FMA should be minimal (typically <= 4 for 32 inner products)
    assert metrics_fma_nofma["max_rel_err"] < 1e-5
    assert metrics_fma_nofma["max_ulp"] <= 8
    # Scalar and non-FMA AVX2 follow the same math (a*b then +c)
    assert metrics_nofma_scalar["max_rel_err"] < 1e-5
    assert metrics_nofma_scalar["max_ulp"] <= 8


def test_fma_catastrophic_cancellation_characterization():
    """Mathematically characterize FMA vs non-FMA behavior under subtractive cancellation.

    When intermediate products nearly cancel (a * b approx -c), non-FMA incurs
    rounding error on a*b that is large relative to the small difference,
    diverging relative error and ULP distance. However, absolute error remains
    rigorously bounded by eps * max(|a*b|, |c|).
    """
    if not tt.has_fma3():
        pytest.skip("FMA3 not supported on this host")

    # Construct test cases where sum of products produces cancellation
    a_val = 1.00000011920928955078125  # 1 + 2^-23
    b_val = 1.0
    # In float32: a_val * b_val in standard rounding is 1.0000001
    # Adding -1.0 leaves 2^-23 in exact arithmetic
    a_mat = np.array([[a_val, 1000.0], [1.0, 1.0]], dtype=np.float32)
    b_mat = np.array([[b_val, 0.0], [-1.0, 0.0]], dtype=np.float32)

    g = tt.Graph()
    x = g.input([2, 2])
    y = g.input([2, 2])
    z = x @ y

    prog_fma = tt.compile(g, outputs=[z], mode="avx2_fma")
    prog_nofma = tt.compile(g, outputs=[z], mode="avx2_nofma")

    res_fma = prog_fma.run({x: a_mat, y: b_mat}).numpy()
    res_nofma = prog_nofma.run({x: a_mat, y: b_mat}).numpy()

    # Verify absolute error is bounded by float machine epsilon * magnitude
    abs_err = np.abs(res_fma - res_nofma)
    max_mag = np.maximum(np.abs(res_fma), np.abs(res_nofma))
    bound = np.finfo(np.float32).eps * np.maximum(max_mag, 1000.0)
    assert np.all(abs_err <= bound)
