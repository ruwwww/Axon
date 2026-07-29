#pragma once

#include <vector>
#include "axon/core/expected.h"
#include "axon/nn/parameter.h"
#include "axon/runtime/runtime.h"

namespace axon {

class SGD {
public:
    SGD(Runtime& rt, std::vector<Parameter*> params, float lr, float momentum = 0.0f);

    Expected<void> step();
    void zero_grad();

private:
    Runtime& rt_;
    std::vector<Parameter*> params_;
    float lr_;
    float momentum_;
    std::vector<Tensor> momentum_bufs_;
};

} // namespace axon
