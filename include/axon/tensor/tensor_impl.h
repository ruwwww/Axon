#pragma once

#include <cstdint>
#include <memory>
#include "axon/storage/storage.h"
#include "axon/tensor/tensor_type.h"

namespace axon {

using TensorId = uint64_t;

class TensorImpl {
public:
    TensorId id_;
    TensorType type_;
    StoragePtr storage_;
    int64_t storage_offset_ = 0;
    bool requires_grad_ = false;

    TensorImpl(TensorId id, TensorType type, StoragePtr storage, bool requires_grad, int64_t storage_offset)
        : id_(id)
        , type_(std::move(type))
        , storage_(std::move(storage))
        , requires_grad_(requires_grad)
        , storage_offset_(storage_offset) {}
};

using TensorImplPtr = std::shared_ptr<TensorImpl>;

} // namespace axon
