#pragma once

#include <string>
#include "tinytensor/graph/graph.hpp"

namespace tinytensor {

class IRPrinter {
public:
    static std::string to_string(const Graph& graph);
    static std::string to_dot(const Graph& graph);
};

} // namespace tinytensor
