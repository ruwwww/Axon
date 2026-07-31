#pragma once

#include <memory>
#include "axon/storage/storage.h"
#include "axon/tensor/tensor_metadata.h"

namespace axon {

class Allocator {
public:
    StoragePtr allocate(const TensorMetadata& type) {
        auto storage = std::make_shared<Storage>(type.size_bytes());
        return storage;
    }
};

} // namespace axon
