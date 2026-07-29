#pragma once

#include <string>
#include "axon/data/dataset.h"
#include "axon/runtime/runtime.h"

namespace axon {

class CIFAR10 : public Dataset {
public:
    CIFAR10(Runtime& rt, const std::string& path, bool train);
    ~CIFAR10() override;

    size_t size() const override;
    std::pair<Tensor, Tensor> get(size_t index) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace axon
