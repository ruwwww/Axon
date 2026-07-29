#pragma once

#include "axon/nn/module.h"

namespace axon {

class Runtime;

class Flatten : public Module {
public:
    Flatten() = default;

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
};

} // namespace axon
