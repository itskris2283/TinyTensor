import time

from tinytensor._tinytensor import Profiler as _NativeProfiler

__all__ = ["Profiler", "benchmark_function"]


class Profiler:
    """TinyTensor runtime and operator execution profiler."""

    def __init__(self):
        self._native = _NativeProfiler()

    @property
    def total_time_us(self) -> float:
        return self._native.total_time_us

    def summary(self) -> str:
        return (
            "=== TinyTensor Execution Profile ===\n"
            f"  Total Duration: {self.total_time_us:.2f} us\n"
            "===================================="
        )

    def __repr__(self) -> str:
        return self.summary()


def benchmark_function(fn, warmup: int = 5, iterations: int = 50):
    """Run warmup and multiple iterations, returning list of execution times in milliseconds."""
    for _ in range(warmup):
        fn()

    times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return times_ms
