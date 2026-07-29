#pragma once

#include "axon/nn/module.h"
#include "axon/nn/parameter.h"

namespace axon {

class Runtime;

class Embedding : public Module {
public:
    Embedding(Runtime& rt, size_t num_embeddings, size_t embedding_dim);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;

private:
    Parameter weight_;
};

} // namespace axon
