#pragma once

#include <cstdint>
#include <memory>
#include "axon/core/expected.h"
#include "axon/storage/storage.h"
#include "axon/tensor/tensor_impl.h"
#include "axon/tensor/tensor_type.h"

namespace axon {

class Runtime;

using TensorId = uint64_t;

class Tensor {
public:
    Tensor() = default;

    Tensor(TensorType type, StoragePtr storage, bool requires_grad = false, int64_t storage_offset = 0)
        : impl_(std::make_shared<TensorImpl>(next_id(), std::move(type), std::move(storage), requires_grad, storage_offset)) {}

    static Tensor empty(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor zeros(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor ones(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor randn(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);

    TensorId id() const { return impl_ ? impl_->id_ : 0; }
    const TensorType& type() const { return impl_->type_; }
    StoragePtr storage() const { return impl_ ? impl_->storage_ : nullptr; }
    bool requires_grad() const { return impl_ ? impl_->requires_grad_ : false; }

    void set_requires_grad(bool val) { if (impl_) impl_->requires_grad_ = val; }

    int64_t storage_offset() const { return impl_ ? impl_->storage_offset_ : 0; }

    const Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void set_grad(const Tensor& g) { grad_ = std::make_shared<Tensor>(g); }

    bool defined() const { return impl_ && impl_->storage_ && impl_->storage_->data != nullptr; }

    template <typename T>
    T* data() const {
        return static_cast<T*>(impl_->storage_->data) + impl_->storage_offset_;
    }

    static Tensor share_impl(TensorImplPtr impl) {
        Tensor t;
        t.impl_ = std::move(impl);
        return t;
    }

private:
    TensorImplPtr impl_;
    std::shared_ptr<Tensor> grad_;

    static TensorId next_id() {
        static TensorId counter = 0;
        return counter++;
    }
};

} // namespace axon
