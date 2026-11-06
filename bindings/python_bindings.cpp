#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tinytensor/autodiff/gradient.hpp"
#include "tinytensor/codegen/codegen.hpp"
#include "tinytensor/codegen/kernel_cache.hpp"
#include "tinytensor/core/operators.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/ir/printer.hpp"
#include "tinytensor/ir/verifier.hpp"
#include "tinytensor/optimizer/optimizer.hpp"
#include "tinytensor/runtime/executor.hpp"
#include "tinytensor/utils/memory_tracker.hpp"

namespace py = pybind11;
using namespace tinytensor;

// Helper to convert py::array to Tensor
Tensor from_numpy(py::array array) {
    py::buffer_info info = array.request();
    std::vector<int64_t> shape_dims;
    for (auto d : info.shape) {
        shape_dims.push_back(static_cast<int64_t>(d));
    }
    Shape shape(std::move(shape_dims));

    DType dtype;
    if (info.format == py::format_descriptor<float>::format()) {
        dtype = DType::Float32;
    } else if (info.format == py::format_descriptor<double>::format()) {
        dtype = DType::Float64;
    } else {
        throw TypeError("Unsupported NumPy dtype: " + info.format + ". Expected float32 or float64.");
    }

    Tensor t(shape, dtype);
    // Copy data into aligned storage
    std::memcpy(t.raw_data(), info.ptr, t.nbytes());
    return t;
}

// Helper to convert Tensor to py::array with safe capsule lifetime
py::array to_numpy(const Tensor& tensor) {
    Tensor contig = tensor.contiguous();
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t item_sz = contig.itemsize();

    for (size_t i = 0; i < contig.ndim(); ++i) {
        shape.push_back(static_cast<size_t>(contig.shape()[i]));
        strides.push_back(static_cast<size_t>(contig.strides()[i]) * item_sz);
    }

    // Allocate StoragePtr on heap and attach to capsule destructor to guarantee buffer lifetime
    auto* holder = new StoragePtr(contig.storage());
    py::capsule base(holder, [](void* ptr) {
        delete static_cast<StoragePtr*>(ptr);
    });

    if (contig.dtype() == DType::Float32) {
        return py::array(
            py::buffer_info(
                contig.data<float>(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                static_cast<ssize_t>(contig.ndim()),
                shape,
                strides
            ),
            base
        );
    } else {
        return py::array(
            py::buffer_info(
                contig.data<double>(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                static_cast<ssize_t>(contig.ndim()),
                shape,
                strides
            ),
            base
        );
    }
}

PYBIND11_MODULE(_tinytensor, m) {
    m.doc() = "TinyTensor: Research-Grade Tensor Compiler and Runtime";

    // Exception translations
    py::register_exception<TinyTensorException>(m, "TinyTensorError");
    py::register_exception<ShapeError>(m, "ShapeError");
    py::register_exception<DimensionError>(m, "DimensionError");
    py::register_exception<BroadcastError>(m, "BroadcastError");
    py::register_exception<TypeError>(m, "TypeError");
    py::register_exception<MemoryError>(m, "MemoryError");
    py::register_exception<ResourceLimitError>(m, "ResourceLimitError");
    py::register_exception<IRError>(m, "IRError");
    py::register_exception<CompileError>(m, "CompileError");
    py::register_exception<RuntimeExecutionError>(m, "RuntimeExecutionError");

    // DType enum
    py::enum_<DType>(m, "DType")
        .value("float32", DType::Float32)
        .value("float64", DType::Float64)
        .export_values();

    m.attr("float32") = DType::Float32;
    m.attr("float64") = DType::Float64;

    // ExecutionMode enum
    py::enum_<ExecutionMode>(m, "ExecutionMode")
        .value("Interpreted", ExecutionMode::Interpreted)
        .value("CompiledScalar", ExecutionMode::CompiledScalar)
        .value("CompiledAVX2", ExecutionMode::CompiledAVX2)
        .value("CompiledAVX2_FMA", ExecutionMode::CompiledAVX2_FMA)
        .value("CompiledAVX2_NoFMA", ExecutionMode::CompiledAVX2_NoFMA)
        .export_values();

    // TargetISA enum
    py::enum_<TargetISA>(m, "TargetISA")
        .value("Scalar", TargetISA::Scalar)
        .value("AVX2", TargetISA::AVX2)
        .value("AVX2_FMA", TargetISA::AVX2_FMA)
        .export_values();

    // Shape
    py::class_<Shape>(m, "Shape")
        .def(py::init<std::vector<int64_t>>())
        .def("__repr__", &Shape::to_string)
        .def("__eq__", &Shape::operator==)
        .def("numel", &Shape::numel)
        .def("ndim", &Shape::ndim)
        .def_property_readonly("dims", &Shape::dims);

    py::implicitly_convertible<std::vector<int64_t>, Shape>();

    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<Shape, DType>(), py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def(py::init([](const std::vector<int64_t>& shape, DType dtype) {
            return std::make_unique<Tensor>(Shape(shape), dtype);
        }), py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("zeros", [](const Shape& shape, DType dtype) { return Tensor::zeros(shape, dtype); }, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("zeros", [](const std::vector<int64_t>& shape, DType dtype) { return Tensor::zeros(Shape(shape), dtype); }, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("ones", [](const Shape& shape, DType dtype) { return Tensor::ones(shape, dtype); }, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("ones", [](const std::vector<int64_t>& shape, DType dtype) { return Tensor::ones(Shape(shape), dtype); }, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("full", [](const Shape& shape, double val, DType dtype) { return Tensor::full(shape, val, dtype); }, py::arg("shape"), py::arg("val"), py::arg("dtype") = DType::Float32)
        .def_static("full", [](const std::vector<int64_t>& shape, double val, DType dtype) { return Tensor::full(Shape(shape), val, dtype); }, py::arg("shape"), py::arg("val"), py::arg("dtype") = DType::Float32)
        .def_static("scalar", &Tensor::scalar, py::arg("val"), py::arg("dtype") = DType::Float32)
        .def_static("empty", [](const Shape& shape, DType dtype) { return Tensor::empty(shape, dtype); }, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("empty", [](const std::vector<int64_t>& shape, DType dtype) { return Tensor::empty(Shape(shape), dtype); }, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("from_numpy", &from_numpy)
        .def("numpy", &to_numpy)
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape().dims(); })
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
        .def_property_readonly("strides", [](const Tensor& t) { return t.strides().strides(); })
        .def("item", [](const Tensor& t) {
            if (t.dtype() == DType::Float32) return py::cast(t.item<float>());
            return py::cast(t.item<double>());
        })
        .def("view", [](const Tensor& t, const std::vector<int64_t>& shape) { return t.view(Shape(shape)); })
        .def("reshape", [](const Tensor& t, const std::vector<int64_t>& shape) { return t.reshape(Shape(shape)); })
        .def("transpose", &Tensor::transpose)
        .def("permute", &Tensor::permute)
        .def("contiguous", &Tensor::contiguous)
        .def("clone", &Tensor::clone)
        .def("broadcast_to", [](const Tensor& t, const std::vector<int64_t>& shape) { return t.broadcast_to(Shape(shape)); })
        .def("__add__", [](const Tensor& a, const Tensor& b) { return ops::add(a, b); })
        .def("__add__", [](const Tensor& a, double b) { return ops::add(a, Tensor::scalar(b, a.dtype())); })
        .def("__radd__", [](const Tensor& a, double b) { return ops::add(Tensor::scalar(b, a.dtype()), a); })
        .def("__sub__", [](const Tensor& a, const Tensor& b) { return ops::sub(a, b); })
        .def("__sub__", [](const Tensor& a, double b) { return ops::sub(a, Tensor::scalar(b, a.dtype())); })
        .def("__rsub__", [](const Tensor& a, double b) { return ops::sub(Tensor::scalar(b, a.dtype()), a); })
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return ops::mul(a, b); })
        .def("__mul__", [](const Tensor& a, double b) { return ops::mul(a, Tensor::scalar(b, a.dtype())); })
        .def("__rmul__", [](const Tensor& a, double b) { return ops::mul(Tensor::scalar(b, a.dtype()), a); })
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return ops::div(a, b); })
        .def("__truediv__", [](const Tensor& a, double b) { return ops::div(a, Tensor::scalar(b, a.dtype())); })
        .def("__rtruediv__", [](const Tensor& a, double b) { return ops::div(Tensor::scalar(b, a.dtype()), a); })
        .def("__matmul__", [](const Tensor& a, const Tensor& b) { return ops::matmul(a, b); })
        .def_property_readonly("T", [](const Tensor& t) { return t.transpose(0, 1); })
        .def("__repr__", [](const Tensor& t) {
            return "Tensor(shape=" + t.shape().to_string() + ", dtype=" + dtype_to_string(t.dtype()) + ")";
        });

    // Node - lifetimes are managed exclusively by Graph, never delete from Python
    py::class_<Node, std::unique_ptr<Node, py::nodelete>>(m, "Node")
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("name", &Node::name)
        .def_property_readonly("shape", [](const Node& n) { return n.shape().dims(); })
        .def_property_readonly("dtype", &Node::dtype)
        .def("__repr__", [](const Node& n) {
            return "Node(id=" + std::to_string(n.id()) + ", op=" + op_type_to_string(n.op_type()) +
                   ", shape=" + n.shape().to_string() + ")";
        });

    // Graph
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("input", [](Graph& g, const std::vector<int64_t>& shape, DType dtype, const std::string& name) {
            return g.create_input(Shape(shape), dtype, name);
        }, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("name") = "",
           py::return_value_policy::reference)
        .def("constant", &Graph::create_constant, py::arg("value"), py::arg("name") = "",
             py::return_value_policy::reference)
        .def("add", &Graph::add, py::return_value_policy::reference)
        .def("sub", &Graph::sub, py::return_value_policy::reference)
        .def("mul", &Graph::mul, py::return_value_policy::reference)
        .def("div", &Graph::div, py::return_value_policy::reference)
        .def("relu", &Graph::relu, py::return_value_policy::reference)
        .def("sigmoid", &Graph::sigmoid, py::return_value_policy::reference)
        .def("matmul", &Graph::matmul, py::return_value_policy::reference)
        .def("transpose", &Graph::transpose, py::return_value_policy::reference)
        .def("reduce_sum", &Graph::reduce_sum, py::arg("a"), py::arg("axes") = std::vector<int64_t>{}, py::arg("keepdims") = false,
             py::return_value_policy::reference)
        .def("reduce_mean", &Graph::reduce_mean, py::arg("a"), py::arg("axes") = std::vector<int64_t>{}, py::arg("keepdims") = false,
             py::return_value_policy::reference)
        .def("mark_output", &Graph::mark_output)
        .def("to_ir", [](const Graph& g) { return IRPrinter::to_string(g); })
        .def("to_dot", [](const Graph& g) { return IRPrinter::to_dot(g); });

    // Optimization Report
    py::class_<OptimizationReport>(m, "OptimizationReport")
        .def_readonly("original_nodes", &OptimizationReport::original_node_count)
        .def_readonly("optimized_nodes", &OptimizationReport::optimized_node_count)
        .def_readonly("constant_foldings", &OptimizationReport::constant_folding_count)
        .def_readonly("dce_eliminated", &OptimizationReport::dce_eliminated_count)
        .def_readonly("cse_eliminated", &OptimizationReport::cse_eliminated_count)
        .def_readonly("fused_groups", &OptimizationReport::fused_groups_count)
        .def_readonly("total_intermediate_allocations", &OptimizationReport::total_intermediate_allocations)
        .def_readonly("peak_memory_bytes", &OptimizationReport::peak_planned_memory_bytes)
        .def_readonly("reused_buffers", &OptimizationReport::reused_buffers_count)
        .def_readonly("memory_savings_bytes", &OptimizationReport::memory_savings_bytes)
        .def("__repr__", &OptimizationReport::to_string);

    // Profiler
    py::class_<Profiler>(m, "Profiler")
        .def(py::init<>())
        .def_property_readonly("total_time_us", &Profiler::total_time_us);

    // KernelCache
    py::class_<KernelCache>(m, "KernelCache")
        .def(py::init<std::string>(), py::arg("cache_dir") = ".tinytensor_cache")
        .def_property_readonly("hits", &KernelCache::hits)
        .def_property_readonly("misses", &KernelCache::misses)
        .def_property_readonly("cache_dir", &KernelCache::cache_dir)
        .def("clear", &KernelCache::clear);

    // Executor / Program
    py::class_<Executor>(m, "Executor")
        .def(py::init([](const Graph& graph, ExecutionMode mode, bool optimize, const std::string& cache_dir) {
            if (cache_dir.empty()) {
                return std::make_unique<Executor>(graph, mode, optimize, nullptr);
            }
            KernelCache cache(cache_dir);
            return std::make_unique<Executor>(graph, mode, optimize, &cache);
        }), py::arg("graph"), py::arg("mode") = ExecutionMode::CompiledAVX2,
            py::arg("optimize") = true, py::arg("cache_dir") = "")
        .def("run", [](Executor& ex, const std::unordered_map<const Node*, Tensor>& inputs, Profiler* prof) {
            std::vector<Tensor> outs = ex.run(inputs, prof);
            if (outs.size() == 1) {
                return py::cast(outs[0]);
            }
            return py::cast(outs);
        }, py::arg("inputs"), py::arg("profiler") = nullptr)
        .def_property_readonly("optimization_report", &Executor::optimization_report)
        .def_property_readonly("original_ir", &Executor::original_ir)
        .def_property_readonly("optimized_ir", &Executor::optimized_ir)
        .def_property_readonly("generated_source", &Executor::generated_source)
        .def_property_readonly("execution_plan", &Executor::execution_plan);

    // Autodiff
    m.def("compute_gradients", &Autodiff::compute_gradients);

    // Hardware features query
    m.def("has_avx", &simd::has_avx);
    m.def("has_avx2", &simd::has_avx2);
    m.def("has_fma3", &simd::has_fma3);
    m.def("supported_target_isa", &simd::supported_target_isa);
    m.def("supported_kernel_features", &simd::supported_kernel_features);

    // Memory Tracking
    py::class_<ProcessMemoryStats>(m, "ProcessMemoryStats")
        .def_readonly("current_rss_bytes", &ProcessMemoryStats::current_rss_bytes)
        .def_readonly("peak_rss_bytes", &ProcessMemoryStats::peak_rss_bytes);

    m.def("get_process_memory_stats", &get_process_memory_stats);
}
