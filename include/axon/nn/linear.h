#pragma once

#include "axon/nn/module.h"
#include "axon/nn/parameter.h"

namespace axon {

class Runtime;

class Linear : public Module {
public:
    Linear(Runtime& rt, size_t in_features, size_t out_features, bool bias = true);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

private:
    Parameter weight_;
    Parameter bias_;
    bool has_bias_;
};

} // namespace axon
