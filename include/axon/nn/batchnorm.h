#pragma once

#include "axon/nn/module.h"
#include "axon/nn/parameter.h"

namespace axon {

class Runtime;

class BatchNorm : public Module {
public:
    BatchNorm(Runtime& rt, size_t channels, float momentum = 0.9f, float epsilon = 1e-5f);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

private:
    Parameter gamma_;
    Parameter beta_;
    Tensor running_mean_;
    Tensor running_var_;
    float momentum_;
    float epsilon_;
};

} // namespace axon
