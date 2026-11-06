#pragma once

#include <stdexcept>
#include <string>

namespace tinytensor {

class TinyTensorException : public std::runtime_error {
public:
    explicit TinyTensorException(const std::string& message)
        : std::runtime_error(message) {}
};

class ShapeError : public TinyTensorException {
public:
    explicit ShapeError(const std::string& message)
        : TinyTensorException("ShapeError: " + message) {}
};

class DimensionError : public TinyTensorException {
public:
    explicit DimensionError(const std::string& message)
        : TinyTensorException("DimensionError: " + message) {}
};

class BroadcastError : public TinyTensorException {
public:
    explicit BroadcastError(const std::string& message)
        : TinyTensorException("BroadcastError: " + message) {}
};

class TypeError : public TinyTensorException {
public:
    explicit TypeError(const std::string& message)
        : TinyTensorException("TypeError: " + message) {}
};

class MemoryError : public TinyTensorException {
public:
    explicit MemoryError(const std::string& message)
        : TinyTensorException("MemoryError: " + message) {}
};

class ResourceLimitError : public TinyTensorException {
public:
    explicit ResourceLimitError(const std::string& message)
        : TinyTensorException("ResourceLimitError: " + message) {}
};

class IRError : public TinyTensorException {
public:
    explicit IRError(const std::string& message)
        : TinyTensorException("IRError: " + message) {}
};

class CompileError : public TinyTensorException {
public:
    explicit CompileError(const std::string& message)
        : TinyTensorException("CompileError: " + message) {}
};

class RuntimeExecutionError : public TinyTensorException {
public:
    explicit RuntimeExecutionError(const std::string& message)
        : TinyTensorException("RuntimeExecutionError: " + message) {}
};

} // namespace tinytensor
