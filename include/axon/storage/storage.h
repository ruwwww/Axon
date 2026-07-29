#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include "axon/storage/quantization.h"

#if defined(_WIN32)
    #include <malloc.h>
    #define AXON_ALIGNED_ALLOC(size, alignment) _aligned_malloc(size, alignment)
    #define AXON_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define AXON_ALIGNED_ALLOC(size, alignment) std::aligned_alloc(alignment, size)
    #define AXON_ALIGNED_FREE(ptr) std::free(ptr)
#endif

namespace axon {

struct Storage {
    void* data;
    size_t size_bytes;
    size_t alignment;
    QuantizationDescriptor quant;

    Storage() : data(nullptr), size_bytes(0), alignment(alignof(max_align_t)) {}

    Storage(size_t size_bytes, size_t alignment = alignof(max_align_t))
        : data(AXON_ALIGNED_ALLOC(size_bytes ? size_bytes : 1, alignment))
        , size_bytes(size_bytes)
        , alignment(alignment)
        , quant{} {}

    ~Storage() {
        if (data) {
            AXON_ALIGNED_FREE(data);
        }
    }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& other) noexcept
        : data(other.data)
        , size_bytes(other.size_bytes)
        , alignment(other.alignment)
        , quant(other.quant)
    {
        other.data = nullptr;
        other.size_bytes = 0;
    }

    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (data) AXON_ALIGNED_FREE(data);
            data = other.data;
            size_bytes = other.size_bytes;
            alignment = other.alignment;
            quant = other.quant;
            other.data = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }
};

using StoragePtr = std::shared_ptr<Storage>;

} // namespace axon
