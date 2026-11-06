#!/usr/bin/env python3
"""Master Final Verification Script for TinyTensor.

Orchestrates 3 consecutive full verification cycles:
  1. C++ GoogleTest Suite (ctest)
  2. Python Pytest Suite (pytest)
  3. All 6 Runnable Examples in examples/
Captures output logs for each run to artifacts/phase3/test_results/consecutive_runs_{1,2,3}.txt.
Exits with code 0 only if all 3 cycles achieve 100% pass rate.
"""

import os
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(ROOT_DIR, "python")
VENV_PYTEST = "/root/tinytensor_venv/bin/pytest"
VENV_PYTHON = "/root/tinytensor_venv/bin/python3"
BUILD_DIR = os.path.join(ROOT_DIR, "build")
RESULTS_DIR = os.path.join(ROOT_DIR, "artifacts", "phase3", "test_results")

EXAMPLES = [
    "examples/basic_tensor.py",
    "examples/custom_graph.py",
    "examples/matrix_multiplication.py",
    "examples/autodiff_example.py",
    "examples/compiler_optimization.py",
    "examples/neural_network_example.py",
]


def run_command(cmd, cwd=ROOT_DIR, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def run_single_cycle(cycle_num: int) -> tuple[bool, str]:
    log_lines = []
    log_lines.append("============================================================")
    log_lines.append(f"TINYTENSOR FINAL VERIFICATION CYCLE {cycle_num}/3")
    log_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("============================================================\n")

    # Step 1: Ninja build
    log_lines.append("[Step 1/4] Building Release C++ and Python Extension with Ninja...")
    rc, stdout, stderr = run_command(["ninja"], cwd=BUILD_DIR)
    if rc != 0:
        log_lines.append(f"FAILED: Ninja build failed (rc={rc}):\n{stderr}")
        return False, "\n".join(log_lines)
    log_lines.append("PASSED: Ninja build clean.\n")

    # Step 2: C++ GoogleTests
    log_lines.append("[Step 2/4] Running C++ GoogleTest Suite via CTest...")
    rc, stdout, stderr = run_command(["ctest", "--output-on-failure"], cwd=BUILD_DIR)
    log_lines.append(stdout)
    if rc != 0:
        log_lines.append(f"FAILED: CTest failed (rc={rc}):\n{stderr}")
        return False, "\n".join(log_lines)
    log_lines.append("PASSED: 100% C++ GoogleTests passed.\n")

    # Step 3: Python PyTest
    log_lines.append("[Step 3/4] Running Python PyTest Suite...")
    rc, stdout, stderr = run_command(
        [VENV_PYTEST, "-v", "tests/python/"],
        cwd=ROOT_DIR,
        env_extra={"PYTHONPATH": f"{PYTHON_DIR}:{ROOT_DIR}"},
    )
    log_lines.append(stdout)
    if rc != 0:
        log_lines.append(f"FAILED: Pytest failed (rc={rc}):\n{stderr}")
        return False, "\n".join(log_lines)
    log_lines.append("PASSED: 100% Python PyTests passed.\n")

    # Step 4: Run all 6 examples
    log_lines.append("[Step 4/4] Executing all 6 Runnable Examples...")
    for ex in EXAMPLES:
        ex_path = os.path.join(ROOT_DIR, ex)
        rc, stdout, stderr = run_command(
            [VENV_PYTHON, ex_path],
            cwd=ROOT_DIR,
            env_extra={"PYTHONPATH": PYTHON_DIR},
        )
        if rc != 0:
            log_lines.append(f"FAILED: Example {ex} failed (rc={rc}):\n{stderr}")
            return False, "\n".join(log_lines)
        log_lines.append(f"  - {ex}: PASSED")

    log_lines.append(f"\nSUCCESS: Cycle {cycle_num}/3 completed cleanly with 0 defects!\n")
    return True, "\n".join(log_lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("============================================================")
    print("TinyTensor Master Verification Protocol: 3 Consecutive Runs")
    print("============================================================\n")

    all_passed = True
    cycle_summaries = []

    for cycle in range(1, 4):
        print(f"--> Starting Cycle {cycle}/3...")
        start_t = time.time()
        success, log_text = run_single_cycle(cycle)
        elapsed = time.time() - start_t

        log_file = os.path.join(RESULTS_DIR, f"consecutive_runs_{cycle}.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_text)

        status_str = "PASSED" if success else "FAILED"
        print(f"--> Cycle {cycle}/3 {status_str} in {elapsed:.2f}s (Log: {log_file})")
        cycle_summaries.append((cycle, status_str, elapsed, log_file))

        if not success:
            all_passed = False
            break

    print("\n============================================================")
    print("FINAL VERIFICATION SUMMARY")
    print("============================================================")
    for c, s, el, path in cycle_summaries:
        print(f"Cycle {c}: {s:<6} | Duration: {el:.2f}s | Log: {path}")

    if all_passed:
        print("\nALL 3 CONSECUTIVE VERIFICATION CYCLES PASSED WITH ZERO ERRORS.")
        sys.exit(0)
    else:
        print("\nVERIFICATION FAILED ON AT LEAST ONE CYCLE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
