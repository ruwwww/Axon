#include "axon/nn/sequential.h"
#include "axon/runtime/runtime.h"

namespace axon {

void Sequential::add(std::unique_ptr<Module> module) {
    modules_.push_back(std::move(module));
}

Expected<Tensor> Sequential::forward(Runtime& rt, const Tensor& x) {
    Tensor current = x;
    for (auto& m : modules_) {
        auto result = m->forward(rt, current);
        if (!result) return result.error();
        current = result.value();
    }
    return current;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> all;
    for (auto& m : modules_) {
        auto child_params = m->parameters();
        all.insert(all.end(), child_params.begin(), child_params.end());
    }
    return all;
}

const std::vector<Parameter*>& Sequential::parameters() const {
    static std::vector<Parameter*> empty;
    return empty;
}

} // namespace axon
