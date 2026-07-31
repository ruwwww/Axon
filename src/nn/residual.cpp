#include "axon/nn/residual.h"
#include "axon/runtime/runtime.h"

namespace axon {

Residual::Residual(std::unique_ptr<Module> module)
    : module_(std::move(module)) {}

Expected<Tensor> Residual::forward(Runtime& rt, const Tensor& x) {
    auto result = module_->forward(rt, x);
    if (!result) return result.error();

    // F(x) + x (broadcast add: same shape assumed)
    auto out_type = TensorMetadata::contiguous(x.type().shape(), x.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), x.requires_grad());
    auto* o_ptr = out.data<float>();
    auto* r_ptr = result.value().data<const float>();
    auto* x_ptr = x.data<const float>();
    auto n = x.type().numel();
    for (int64_t i = 0; i < n; ++i) o_ptr[i] = r_ptr[i] + x_ptr[i];

    return out;
}

std::vector<Parameter*> Residual::parameters() {
    return module_->parameters();
}

} // namespace axon
