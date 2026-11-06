cmake_minimum_required(VERSION 3.25)
project(TinyTensor
    VERSION 0.1.1
    DESCRIPTION "Research-grade tensor compiler, numerical runtime and automatic differentiation framework"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_TESTING "Build test suite" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

# Warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core library source files
set(TINYTENSOR_SRCS
    src/core/tensor.cpp
    src/core/operators.cpp
    src/utils/errors.cpp
    src/utils/memory_tracker.cpp
    src/runtime/executable.cpp
    src/runtime/executor.cpp
    src/runtime/interpreter.cpp
    src/runtime/profiler.cpp
    src/graph/node.cpp
    src/graph/graph.cpp
    src/ir/operation.cpp
    src/ir/ir.cpp
    src/ir/verifier.cpp
    src/ir/printer.cpp
    src/autodiff/gradient.cpp
    src/autodiff/backward.cpp
    src/optimizer/optimizer.cpp
    src/optimizer/constant_folding.cpp
    src/optimizer/dead_code_elimination.cpp
    src/optimizer/cse.cpp
    src/optimizer/fusion.cpp
    src/optimizer/memory_planner.cpp
    src/codegen/codegen.cpp
    src/codegen/cpu_features.cpp
    src/codegen/cpp_backend.cpp
    src/codegen/kernel_cache.cpp
)

# AVX2 backend compiled strictly with AVX2/FMA flags in an isolated translation unit
set(TINYTENSOR_AVX2_SRCS
    src/codegen/avx2_backend.cpp
)

add_library(tinytensor_core STATIC
    ${TINYTENSOR_SRCS}
    ${TINYTENSOR_AVX2_SRCS}
)

target_include_directories(tinytensor_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tinytensor_core PUBLIC
    ${CMAKE_DL_LIBS}
)

# Set AVX2 flags only for the AVX2 translation unit
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${TINYTENSOR_AVX2_SRCS} PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma"
    )
endif()

# Python bindings
if(BUILD_PYTHON)
    execute_process(
        COMMAND /root/tinytensor_venv/bin/python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_FOUND_RET
    )
    if(pybind11_FOUND_RET EQUAL 0 AND EXISTS "${pybind11_DIR}")
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        find_package(pybind11 REQUIRED)
    endif()

    pybind11_add_module(_tinytensor
        bindings/python_bindings.cpp
    )
    target_link_libraries(_tinytensor PRIVATE tinytensor_core)
    target_include_directories(_tinytensor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    if(NOT ENABLE_ASAN AND NOT ENABLE_COVERAGE)
        set_target_properties(_tinytensor PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/tinytensor"
        )
    endif()
endif()

# C++ Testing with GoogleTest
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)

    # Core tensor & operators unit tests
    add_executable(test_core
        tests/cpp/core/test_tensor.cpp
        tests/cpp/core/test_shape_strides.cpp
        tests/cpp/core/test_broadcasting.cpp
        tests/cpp/core/test_storage.cpp
        tests/cpp/core/test_operators.cpp
        tests/cpp/core/test_numerical_edge_cases.cpp
        tests/cpp/core/test_core_branch_expansion.cpp
    )
    target_link_libraries(test_core PRIVATE tinytensor_core GTest::GTest GTest::Main)
    add_test(NAME CoreTests COMMAND test_core)

    # Graph and IR unit tests
    add_executable(test_graph_ir
        tests/cpp/graph/test_graph.cpp
        tests/cpp/ir/test_ir.cpp
        tests/cpp/ir/test_verifier.cpp
        tests/cpp/ir/test_verifier_branch_expansion.cpp
    )
    target_link_libraries(test_graph_ir PRIVATE tinytensor_core GTest::GTest GTest::Main)
    add_test(NAME GraphIRTests COMMAND test_graph_ir)

    # Autodiff tests
    add_executable(test_autodiff
        tests/cpp/autodiff/test_autodiff.cpp
        tests/cpp/autodiff/test_autodiff_branch_expansion.cpp
    )
    target_link_libraries(test_autodiff PRIVATE tinytensor_core GTest::GTest GTest::Main)
    add_test(NAME AutodiffTests COMMAND test_autodiff)

    # Optimizer tests
    add_executable(test_optimizer
        tests/cpp/optimizer/test_optimizer.cpp
        tests/cpp/optimizer/test_adversarial_memory_planner.cpp
    )
    target_link_libraries(test_optimizer PRIVATE tinytensor_core GTest::GTest GTest::Main)
    add_test(NAME OptimizerTests COMMAND test_optimizer)

    # Runtime, Codegen & SIMD tests
    add_executable(test_runtime_codegen
        tests/cpp/runtime/test_interpreter.cpp
        tests/cpp/runtime/test_executor.cpp
        tests/cpp/codegen/test_codegen.cpp
        tests/cpp/codegen/test_avx2.cpp
        tests/cpp/codegen/test_cpu_features.cpp
        tests/cpp/codegen/test_kernel_cache.cpp
        tests/cpp/codegen/test_compiler_failure_recovery.cpp
        tests/cpp/codegen/test_codegen_branch_expansion.cpp
    )
    target_link_libraries(test_runtime_codegen PRIVATE tinytensor_core GTest::GTest GTest::Main)
    add_test(NAME RuntimeCodegenTests COMMAND test_runtime_codegen)
endif()
