#pragma once

#include <vector>
#include "axon/core/expected.h"
#include "axon/nn/parameter.h"
#include "axon/runtime/runtime.h"

namespace axon {

class AdamW {
public:
    AdamW(Runtime& rt, std::vector<Parameter*> params, float lr,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
          float weight_decay = 0.01f);

    Expected<void> step();
    void zero_grad();

private:
    struct State {
        Tensor m;
        Tensor v;
    };

    Runtime& rt_;
    std::vector<Parameter*> params_;
    std::vector<State> state_;
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int64_t t_ = 0;
};

} // namespace axon
