# TinyTensor Environment & Feasibility Audit (Phase 0)

**Date:** 2026-11-05  
**Auditor:** Principal Systems Engineer / Compiler Architect  
**Project:** TinyTensor v0.1.0  
**Target Architecture:** x86-64 / AVX2  
**Host Environment:** Windows 11 Enterprise (Lenovo ThinkPad, Intel Core i7-8650U @ 1.90GHz, 4 cores, 8 logical processors, 16 GB Physical RAM)  
**Primary Execution Platform:** Ubuntu 24.04.1 LTS on WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2 x86_64)

---

## 1. System Inventory

| Component | Specification / Version Detected | Requirement Status |
| :--- | :--- | :--- |
| **Operating System** | Ubuntu 24.04.1 LTS (Noble Numbat), kernel 6.18.54.1-microsoft-standard-WSL2 | COMPLIANT (Target: Ubuntu 24.04 LTS / WSL2) |
| **CPU Architecture** | x86_64, 4 physical cores, 8 threads | COMPLIANT |
| **SIMD Support** | AVX, AVX2, FMA, BMI1, BMI2, SSE4.2, SSE4.1 | COMPLIANT (Target: AVX2 with scalar fallback) |
| **System Memory (RAM)** | 16 GB Physical Host (17,011,474,432 bytes); WSL assigned 7.86 GB (7.4 GB available) | COMPLIANT (Target: 16 GB system RAM) |
| **Storage / Disk Space** | 952 GB available on `/` (ext4) and `/mnt/d` (NTFS mount) | COMPLIANT |
| **C++ Compilers** | GCC 13.3.0 (`/usr/bin/g++`), Clang 18.1.3 (`/usr/bin/clang++`) | COMPLIANT (GCC 13+ / Clang 18+) |
| **Build System** | CMake 3.28.3, Ninja 1.11.1 | COMPLIANT (CMake 3.25+, Ninja) |
| **Python Runtime** | Python 3.12.3 (`/usr/bin/python3`) | COMPLIANT (Python 3.12) |
| **Testing Harness** | GoogleTest 1.14.0 (`libgtest-dev`), pytest 8.x | COMPLIANT |
| **Coverage Tools** | gcovr 7.0, pytest-cov | COMPLIANT |
| **Linters & Formatters** | clang-format-18, clang-tidy-18, ruff | COMPLIANT |
| **Sanitizers** | ASan (`libclang-rt-18-dev`), UBSan (`-fsanitize=address,undefined`) | COMPLIANT |

---

## 2. CPU Feature Verification (AVX2 & FMA)

Inspection via `lscpu` and `/proc/cpuinfo` confirms:
```text
Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36
       clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm arch_perfmon
       rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma
       cx16 pdcm pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand
       hypervisor lahf_lm abm 3dnowprefetch pti ssbd ibrs ibpb stibp tpr_shadow
       ept vpid ept_ad fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed
       adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves vnmi md_clear flush_l1d
```
Key ISA extensions present:
- `avx`: Advanced Vector Extensions (256-bit FP vectors)
- `avx2`: Advanced Vector Extensions 2 (256-bit integer and floating-point vector instructions)
- `fma`: Fused Multiply-Accumulate ($a \cdot b + c$) for high-throughput linear algebra
- `bmi1`, `bmi2`: Bit manipulation instructions

---

## 3. Dependency Installation Strategy

1. **C++ Toolchain & Core Libraries:**
   - Compiler: GCC 13.3.0 and Clang 18.1.3 native Ubuntu packages.
   - Build Tools: CMake 3.28.3 and Ninja 1.11.1.
   - GoogleTest: Canonical Ubuntu package `libgtest-dev` (version 1.14.0) with CMake `find_package(GTest REQUIRED)` or FetchContent fallback.
   - pybind11: Installed in Python virtual environment (`.venv`) and discovered via `pybind11Config.cmake` / `python3 -m pybind11 --cmake`.

2. **Python Environment:**
   - Isolated `.venv` (Python 3.12.3) containing:
     - `pybind11`
     - `numpy` (correctness reference)
     - `torch` CPU wheel (differential validation)
     - `pytest`, `pytest-cov`
     - `ruff`
     - `matplotlib`

3. **Compiler Backend Isolation:**
   - Generated native kernels compiled dynamically using system `g++` / `clang++` via C++ standard process execution without invoking an external shell or shell string interpolation.
   - No hard dependency on LLVM JIT library at runtime; generated C++ uses the C ABI and standard dynamic library loader (`dlopen` / `dlsym`).

---

## 4. Feasibility & Risk Assessment

1. **WSL Filesystem I/O:**
   - Working in `/mnt/d/MERN/TinyTensor` provides direct visibility across Windows and WSL.
   - Dynamic library compilation (`.so`) is verified on Linux filesystem paths.
2. **AVX2 Dispatch Safety:**
   - The runtime implements hardware feature detection using `__builtin_cpu_supports("avx2")` and `cpuid`.
   - Never globally apply `-mavx2` to the entire codebase. The base library compiles with `-march=x86-64`, and AVX2 translation units are compiled with `-mavx2 -mfma` in isolated compilation units with automatic fallback.
3. **Reproducibility:**
   - All random seeds, tolerances, and test inputs are strictly fixed and deterministic.
