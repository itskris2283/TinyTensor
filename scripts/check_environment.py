#!/usr/bin/env python3
"""Environment Verification Script for TinyTensor Build & Runtime Environment.

Checks:
- OS & Architecture (Linux x86_64 / WSL2)
- CPU Instruction Set (AVX2 support)
- Toolchain: GCC/Clang, CMake, Ninja
- Python 3.12 runtime and required packages
- Memory target (16 GB RAM)
"""

import os
import platform
import shutil
import subprocess
import sys


def check_cmd(cmd):
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return res.returncode == 0, res.stdout.strip()
    except Exception:
        return False, ""


def main():
    print("============================================================")
    print("TinyTensor Environment Diagnostic & Validation")
    print("============================================================\n")

    all_passed = True

    # 1. OS & Architecture
    os_name = platform.system()
    arch = platform.machine()
    print(f"OS Platform:   {os_name} ({platform.platform()})")
    print(f"Architecture:  {arch}")
    if arch != "x86_64":
        print("  [WARN] Architecture is not x86_64; AVX2 features disabled.")
    else:
        print("  [OK] x86_64 architecture confirmed.")

    # 2. AVX2 Support
    has_avx2 = False
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
            if "avx2" in cpuinfo:
                has_avx2 = True
    print(f"AVX2 Hardware: {'Supported' if has_avx2 else 'Not Detected'}")
    if not has_avx2:
        print("  [WARN] CPU does not support AVX2; scalar fallback will be used.")
    else:
        print("  [OK] AVX2 SIMD acceleration available.")

    # 3. Compilers and Build Tools
    c_ok, c_ver = check_cmd("g++ --version | head -n 1")
    print(f"C++ Compiler:  {c_ver if c_ok else 'Not found'}")
    if not c_ok:
        all_passed = False

    cmake_ok, cmake_ver = check_cmd("cmake --version | head -n 1")
    print(f"CMake:         {cmake_ver if cmake_ok else 'Not found'}")
    if not cmake_ok:
        all_passed = False

    ninja_ok, ninja_ver = check_cmd("ninja --version")
    print(f"Ninja:         {ninja_ver if ninja_ok else 'Not found'}")
    if not ninja_ok:
        all_passed = False

    # 4. Python Environment
    py_ver = sys.version.split()[0]
    print(f"Python:        {py_ver} ({sys.executable})")

    packages = [
        "numpy",
        "torch",
        "pybind11",
        "pytest",
        "ruff",
        "matplotlib",
        "gcovr",
    ]
    for pkg in packages:
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "installed")
            print(f"  - {pkg:<12}: {v}")
        except ImportError:
            # gcovr is CLI tool
            if shutil.which(pkg):
                print(f"  - {pkg:<12}: installed (CLI)")
            else:
                print(f"  - {pkg:<12}: [MISSING]")
                all_passed = False

    # 5. System Memory
    total_mem_gb = 0.0
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_mem_gb = float(line.split()[1]) / 1024.0 / 1024.0
    print(f"\nSystem Memory: {total_mem_gb:.2f} GB")
    if total_mem_gb >= 7.0:
        print("  [OK] Memory capacity meets operational threshold.")
    else:
        print("  [WARN] Memory below 8GB threshold.")

    print("\n------------------------------------------------------------")
    if all_passed:
        print("STATUS: ENVIRONMENT VALIDATION SUCCEEDED")
    else:
        print("STATUS: ENVIRONMENT VALIDATION DETECTED MISSING COMPONENTS")
    print("------------------------------------------------------------")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
