#!/usr/bin/env python3
"""Bounded test orchestration script for TinyTensor test suite.

Runs both C++ (GoogleTest via CTest) and Python (pytest) suites with bounded
timeouts.
"""

import os
import signal
import subprocess
import sys
import time


def run_bounded_command(cmd, cwd=None, timeout_sec=120, env=None):
    print(f"--> Running: {' '.join(cmd)}")
    start_time = time.perf_counter()

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # Create distinct process group
    )

    try:
        stdout, _ = proc.communicate(timeout=timeout_sec)
        elapsed = time.perf_counter() - start_time
        retcode = proc.returncode
        return retcode, stdout, elapsed
    except subprocess.TimeoutExpired:
        print(f"!!! Command timed out after {timeout_sec}s, killing process group...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(1)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
        return -1, stdout, timeout_sec


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(root_dir, "build")
    python_dir = os.path.join(root_dir, "python")

    print("========================================")
    print("TinyTensor Bounded Verification Suite")
    print("========================================")

    # 1. C++ Test Suite
    ctest_cmd = ["ctest", "--test-dir", build_dir, "--output-on-failure", "-j", "4"]
    ret_cpp, out_cpp, elapsed_cpp = run_bounded_command(ctest_cmd, cwd=root_dir, timeout_sec=90)
    print(out_cpp)
    if ret_cpp != 0:
        print(f"FAILED: C++ test suite exited with code {ret_cpp}")
        sys.exit(1)
    print(f"PASSED: C++ tests completed in {elapsed_cpp:.2f}s\n")

    # 2. Python Test Suite
    pytest_bin = "/root/tinytensor_venv/bin/pytest"
    pytest_cmd = [pytest_bin, "-v", "--tb=short", "tests/python"]
    env = {"PYTHONPATH": f"{python_dir}:{root_dir}"}
    ret_py, out_py, elapsed_py = run_bounded_command(
        pytest_cmd, cwd=root_dir, timeout_sec=300, env=env
    )
    print(out_py)
    if ret_py != 0:
        print(f"FAILED: Python test suite exited with code {ret_py}")
        sys.exit(1)
    print(f"PASSED: Python tests completed in {elapsed_py:.2f}s\n")

    print("========================================")
    print("ALL TEST SUITES PASSED SUCCESSFULLY!")
    print("========================================")


if __name__ == "__main__":
    main()
