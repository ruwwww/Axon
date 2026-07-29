#pragma once

#include "axon/nn/module.h"
#include "axon/nn/parameter.h"

namespace axon {

class Runtime;

class Conv2D : public Module {
public:
    Conv2D(Runtime& rt, size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, bool bias = true);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

private:
    Parameter weight_;
    Parameter bias_;
    size_t stride_, padding_;
    bool has_bias_;
};

} // namespace axon
