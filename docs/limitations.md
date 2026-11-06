# TinyTensor Known Limitations and Future Roadmap

## 1. Architectural Scope and Intentional Trade-offs

TinyTensor is designed as a lightweight, clean, research-grade tensor compiler, educational reference runtime, and autodiff framework. To avoid dependency sprawl and preserve architectural clarity:
- **No LLVM Dependency:** TinyTensor generates C++ source files compiled via system `g++`/`clang++` rather than generating LLVM bitcode. This simplifies debugging and deployment, though runtime compilation incurs a sub-second invocation overhead on cache misses.
- **CPU Focus:** TinyTensor v0.1.0 targets x86-64 CPU architecture with AVX2 SIMD acceleration. GPU (CUDA/ROCm) execution is reserved for future versions.
- **Float32 & Float64 Only:** Quantized integer data types (int8, int16, bfloat16, fp16) are not currently supported in v0.1.0.

## 2. Operator Coverage Boundaries

While TinyTensor supports full end-to-end multi-layer perceptron training and composite non-linear workloads, certain specialized neural network layers are intentionally deferred to future releases:
- Convolutional operators (`Conv1D`, `Conv2D`, `ConvTranspose`) are not natively implemented in v0.1.0.
- Dynamic shapes: Shape dimensions must be statically known or inferred at graph compilation time.
- Control flow operations (dynamic loops `while`, conditionals `if-else`) are executed at the host Python layer rather than inside the compiled DAG.

## 3. Future Roadmap

1. **v0.2.0:**
   - Multi-threaded CPU execution engine via thread pool.
   - Dynamic shape inference with symbolic dimensions.
   - 2D Spatial Convolution with im2col and Winograd transformations.
2. **v0.3.0:**
   - Optional LLVM JIT backend (`llvm::ExecutionEngine`) as an alternative to `g++` fork/exec.
   - Float16 and BFloat16 numerical representations.
   - CUDA / OpenCL GPU kernel generation backend.
