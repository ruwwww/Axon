#include "axon/nn/embedding.h"
#include "axon/runtime/runtime.h"

namespace axon {

Embedding::Embedding(Runtime& rt, size_t num_embeddings, size_t embedding_dim)
    : weight_(Tensor::randn(rt, {static_cast<int64_t>(num_embeddings), static_cast<int64_t>(embedding_dim)}), true)
{
    register_parameter("weight", &weight_);
}

Expected<Tensor> Embedding::forward(Runtime& rt, const Tensor& x) {
    const auto& indices = x;
    auto N = indices.type().numel();
    auto embedding_dim = weight_.tensor().type().shape()[1];

    auto out_type = TensorType::contiguous({N, embedding_dim}, weight_.tensor().type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), true);

    auto* idx_ptr = indices.data<const int64_t>();
    auto* w_ptr = weight_.tensor().data<const float>();
    auto* o_ptr = out.data<float>();

    for (int64_t i = 0; i < N; ++i) {
        auto row = idx_ptr[i];
        for (int64_t j = 0; j < embedding_dim; ++j) {
            o_ptr[i * embedding_dim + j] = w_ptr[row * embedding_dim + j];
        }
    }

    return out;
}

} // namespace axon
