#pragma once

#include <cstddef>
#include <cstdint>
#include "axon/core/types.h"

namespace axon {

struct QuantizationDescriptor {
    QuantFormat format = QuantFormat::None;
    size_t block_size = 0;
};

} // namespace axon
