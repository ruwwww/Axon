#pragma once

#include <memory>
#include <vector>
#include "axon/nn/module.h"

namespace axon {

class Runtime;

class Sequential : public Module {
public:
    void add(std::unique_ptr<Module> module);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

    std::vector<Parameter*> parameters() override;

    const std::vector<Parameter*>& parameters() const override;

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace axon
