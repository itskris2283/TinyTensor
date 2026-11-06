# TinyTensor Kernel Generation Engine

## 1. Code Synthesis Architecture

TinyTensor generates C++20 code on-the-fly rather than relying on an external heavy compiler framework like LLVM. The `tinytensor::CppBackend` synthesizes clean, human-readable, self-contained translation units (.cpp files) with the following structure:

```cpp
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

extern "C" {

void tinytensor_kernel(const float* const* inputs,
                       float* const* outputs,
                       float* const* intermediates) {
    // Aligned pointers and scalar/vector execution loops
}

}
```

## 2. Synthesis of Operators

### 2.1 Elementwise Arithmetic & Fusion
For fused or elementary operations, loops are emitted either linearly over flat contiguous arrays or via multi-dimensional index mapping for strided/broadcast layouts:
```cpp
// Fused ReLU(A + B)
for (int64_t i = 0; i < N; ++i) {
    float val = in0[i] + in1[i];
    out[i] = val > 0.0f ? val : 0.0f;
}
```

### 2.2 Matrix Multiplication
Emits optimized tiled loops with accumulator registers:
```cpp
for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
    }
}
```

### 2.3 Reductions
Emits general nested multi-dimensional index loops computing coordinates from the flat linear index, accumulating along reduction axes into the output tensor.

## 3. Subprocess Execution & Fork/Exec Security

Kernel compilation is sandboxed out-of-process:
- Uses `fork()` and `execve()` to invoke the host compiler (`g++` or `clang++`).
- No shell injection risk: commands are passed as a discrete `char* argv[]` array, bypassing `system()` and shell interpreters.
- Executed in a distinct POSIX process group with strict timeouts (`setpgid()`, `SIGKILL` on timeout) ensuring no runaway compiler processes can ever hang the system.
- Flags used:
  `-O3 -std=c++20 -fPIC -shared -march=native -Wall -Wextra -fno-math-errno`
