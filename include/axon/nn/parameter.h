#pragma once

#include <cstring>
#include "axon/tensor/tensor.h"

namespace axon {

class Parameter {
public:
    Parameter() = default;
    Parameter(Tensor tensor, bool trainable = true)
        : tensor_(std::move(tensor)), trainable_(trainable) {}

    Tensor& tensor() { return tensor_; }
    const Tensor& tensor() const { return tensor_; }

    Tensor& grad() {
        if (!grad_) {
            auto type = TensorType::contiguous(tensor_.type().shape(), tensor_.type().dtype());
            auto storage = std::make_shared<Storage>(type.size_bytes());
            grad_ = std::make_unique<Tensor>(std::move(type), std::move(storage), false);
            memset(grad_->storage()->data, 0, grad_->storage()->size_bytes);


        }
        return *grad_;
    }

    bool has_grad() const { return grad_ != nullptr; }

    bool trainable() const { return trainable_; }
    void set_trainable(bool val) { trainable_ = val; }

private:
    Tensor tensor_;
    std::unique_ptr<Tensor> grad_;
    bool trainable_ = true;
};

} // namespace axon
