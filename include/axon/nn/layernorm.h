#pragma once

#include "axon/nn/module.h"
#include "axon/nn/parameter.h"
#include <cstdint>
#include <vector>

namespace axon {

class Runtime;

class LayerNorm : public Module {
public:
    LayerNorm(Runtime& rt, const std::vector<int64_t>& normalized_shape, float epsilon = 1e-5f);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

private:
    Parameter gamma_;
    Parameter beta_;
    int64_t normalized_size_;
    float epsilon_;
};

} // namespace axon
