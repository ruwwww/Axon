#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include "axon/core/expected.h"
#include "axon/storage/storage.h"
#include "axon/tensor/tensor_type.h"

namespace axon {

class Runtime;

using TensorId = uint64_t;

class Tensor {
public:
    Tensor() = default;

    Tensor(TensorType type, StoragePtr storage, bool requires_grad = false, int64_t storage_offset = 0)
        : id_(next_id())
        , type_(std::move(type))
        , storage_(std::move(storage))
        , requires_grad_(requires_grad)
        , storage_offset_(storage_offset) {}

    static Tensor empty(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor zeros(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor ones(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor randn(Runtime& rt, const std::vector<int64_t>& shape, DType dtype = DType::Float32);

    TensorId id() const { return id_; }
    const TensorType& type() const { return type_; }
    StoragePtr storage() const { return storage_; }
    bool requires_grad() const { return requires_grad_; }

    void set_requires_grad(bool val) { requires_grad_ = val; }

    int64_t storage_offset() const { return storage_offset_; }

    const Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void set_grad(const Tensor& g) { grad_ = std::make_shared<Tensor>(g); }

    bool defined() const { return storage_ != nullptr && storage_->data != nullptr; }

    template <typename T>
    T* data() const {
        return static_cast<T*>(storage_->data) + storage_offset_;
    }

private:
    static TensorId next_id() {
        static TensorId counter = 0;
        return counter++;
    }

    TensorId id_ = 0;
    TensorType type_;
    StoragePtr storage_;
    int64_t storage_offset_ = 0;
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
};

} // namespace axon
