#!/usr/bin/env python3
"""Run TinyTensor benchmark suite with bounded execution timeouts."""

import os
import subprocess
import sys


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bench_script = os.path.join(root_dir, "benchmarks", "run_benchmarks.py")
    python_bin = sys.executable

    print(f"--> Launching benchmark orchestration: {bench_script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{os.path.join(root_dir, 'python')}:{os.path.join(root_dir, 'benchmarks')}:{root_dir}"
    )

    proc = subprocess.run([python_bin, bench_script], cwd=root_dir, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
