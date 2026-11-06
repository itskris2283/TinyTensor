# TinyTensor Developer and Contributor Guide

## 1. Prerequisites and Toolchain

Building TinyTensor from scratch requires:
- Linux x86-64 / WSL2 (Ubuntu 24.04 LTS recommended)
- C++20 compliant compiler: GCC 13+ or Clang 16+
- CMake $\ge 3.25$
- Ninja build system
- Python 3.12 with development headers (`python3-dev`)
- Virtual environment with: `numpy`, `torch`, `pybind11`, `pytest`, `pytest-cov`, `ruff`, `matplotlib`, `gcovr`

## 2. Build Configurations

### 2.1 Release Build (Default)
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

### 2.2 Sanitizer Build (AddressSanitizer & UndefinedBehaviorSanitizer)
```bash
cmake -B build_asan -G Ninja -DENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
ninja -C build_asan
ctest --test-dir build_asan --output-on-failure
```

### 2.3 Coverage Build
```bash
cmake -B build_cov -G Ninja -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
ninja -C build_cov
ctest --test-dir build_cov --output-on-failure
gcovr --root . --filter 'src/' --exclude 'src/codegen/cpp_backend.cpp' --print-summary
```

## 3. Running Test Suites

TinyTensor enforces bounded execution timeouts:
```bash
# Run both C++ GoogleTest and Python pytest suites
python3 scripts/run_all_tests.py
```

## 4. Code Quality and Formatting

```bash
# Lint Python codebase
ruff check python/ tests/python/ scripts/ examples/ benchmarks/

# Format C++ codebase
clang-format -i include/tinytensor/**/*.hpp src/**/*.cpp tests/cpp/**/*.cpp bindings/*.cpp
```

## 5. Adding a New Operator to TinyTensor

Adding a new operation (e.g. `Tanh`) requires:
1. `include/tinytensor/ir/operation.hpp`: Add `OpType::Tanh` to the enum and `op_type_to_string()`.
2. `src/core/operators.cpp`: Implement eager numerical evaluation (`ops::tanh`).
3. `src/graph/graph.cpp`: Implement `Graph::tanh(Node* a)` with shape and dtype propagation.
4. `src/ir/verifier.cpp`: Add verification rules (arity = 1, dtype preservation).
5. `src/autodiff/gradient.cpp`: Implement gradient derivation: $\bar{x} = \bar{z} \cdot (1 - z^2)$.
6. `src/codegen/cpp_backend.cpp`: Emit C++ kernel loop (`std::tanh` or AVX2 approximation).
7. `bindings/python_bindings.cpp`: Expose operator to Python `Graph` and `Tensor`.
8. `tests/`: Add unit tests in C++ (`tests/cpp/`) and Python (`tests/python/`).
