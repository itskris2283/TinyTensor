#pragma once

#include "tinytensor/graph/graph.hpp"

namespace tinytensor {

class IRVerifier {
public:
    // Verifies full IR semantics. Throws IRError on any violation.
    static void verify(const Graph& graph);
};

} // namespace tinytensor
