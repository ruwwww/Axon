#pragma once

#include "axon/nn/module.h"

namespace axon {

class Runtime;

class Dropout : public Module {
public:
    Dropout(float p = 0.5f);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

private:
    float p_;
};

} // namespace axon
