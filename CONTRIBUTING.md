# Contributing to TinyTensor

Thank you for your interest in contributing to TinyTensor!

## 1. Development Environment

Follow [docs/development.md](docs/development.md) to set up your local development environment using Ubuntu 24.04 LTS / WSL2 with GCC 13+, CMake 3.25+, Ninja, and Python 3.12.

## 2. Pull Request & Coding Guidelines

1. **Maintain Architectural Independence:** TinyTensor must never wrap or call NumPy, PyTorch, or Eigen internally for execution. Reference libraries are reserved for differential correctness tests only.
2. **Strict Hygiene:**
   - C++ code must compile with zero warnings under:
     `-Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion -Wshadow -Wnon-virtual-dtor -std=c++20`.
   - Format C++ files with `clang-format`.
   - Lint Python code with `ruff check`.
3. **Safety and Sanitizers:**
   - Any new C++ logic must pass AddressSanitizer and UndefinedBehaviorSanitizer without errors or memory leaks (`cmake -B build_asan -DENABLE_ASAN=ON`).
4. **Test Coverage Standards:**
   - Maintain at least **85% line coverage** on C++ core logic (`build_cov`).
   - Maintain at least **90% line coverage** on Python modules (`pytest --cov=tinytensor`).
5. **Bounded Execution:** Tests and benchmarks must include timeouts and clean process management.
