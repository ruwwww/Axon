#include "axon/nn/flatten.h"
#include "axon/runtime/runtime.h"

namespace axon {

Expected<Tensor> Flatten::forward(Runtime& rt, const Tensor& x) {
    const auto& shape = x.type().shape();
    if (shape.empty()) return Error{"Flatten: empty input"};

    int64_t batch = shape[0];
    int64_t flattened = 1;
    for (size_t i = 1; i < shape.size(); ++i) flattened *= shape[i];

    auto out_type = TensorMetadata::contiguous({batch, flattened}, x.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), x.requires_grad());

    auto* x_ptr = x.data<const float>();
    auto* o_ptr = out.data<float>();
    auto n = x.type().numel();
    for (int64_t i = 0; i < n; ++i) o_ptr[i] = x_ptr[i];

    return out;
}

} // namespace axon
