#pragma once

#include <string>
#include <vector>
#include "axon/core/expected.h"
#include "axon/nn/parameter.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime;

class Module {
public:
    virtual ~Module() = default;

    void register_parameter(const std::string& name, Parameter* param);
    virtual std::vector<Parameter*> parameters() { return parameters_; }
    virtual const std::vector<Parameter*>& parameters() const { return parameters_; }
    const std::vector<std::string>& parameter_names() const { return param_names_; }

    virtual Expected<Tensor> forward(Runtime& rt, const Tensor& x) = 0;

    void train() { training_ = true; }
    void eval() { training_ = false; }
    bool is_training() const { return training_; }

private:
    std::vector<Parameter*> parameters_;
    std::vector<std::string> param_names_;
    bool training_ = true;
};

} // namespace axon
