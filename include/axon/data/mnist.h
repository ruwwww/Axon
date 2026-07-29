#pragma once

#include <string>
#include "axon/data/dataset.h"
#include "axon/runtime/runtime.h"

namespace axon {

class MNIST : public Dataset {
public:
    MNIST(Runtime& rt, const std::string& path, bool train);
    ~MNIST() override;

    size_t size() const override;
    std::pair<Tensor, Tensor> get(size_t index) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace axon
