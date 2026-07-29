#pragma once

#include <memory>
#include "axon/nn/module.h"

namespace axon {

class Runtime;

class Residual : public Module {
public:
    Residual(std::unique_ptr<Module> module);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

    std::vector<Parameter*> parameters() override;

private:
    std::unique_ptr<Module> module_;
};

} // namespace axon
