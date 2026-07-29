#include "axon/nn/module.h"

namespace axon {

void Module::register_parameter(const std::string& name, Parameter* param) {
    parameters_.push_back(param);
    param_names_.push_back(name);
}

} // namespace axon
