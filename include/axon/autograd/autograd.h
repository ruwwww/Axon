#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include "axon/autograd/node.h"
#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime;

class Graph {
public:
    void append(std::shared_ptr<Node> node);
    void clear() { nodes_.clear(); }
    size_t size() const;
    const std::shared_ptr<Node>& operator[](size_t i) const;
    std::shared_ptr<Node>& operator[](size_t i);
    const std::vector<std::shared_ptr<Node>>& nodes() const { return nodes_; }

private:
    std::vector<std::shared_ptr<Node>> nodes_;
};

struct MatMulOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& a, const Tensor& b);
};

struct ReLUOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x);
};

struct AddOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& a, const Tensor& b);
};

struct Conv2DOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input, const Tensor& weight,
                                    const Tensor& bias, int64_t stride, int64_t padding);
};

struct MaxPool2dOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input, int64_t kernel, int64_t stride);
};

struct AvgPool2dOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input, int64_t kernel, int64_t stride);
};

struct BatchNormOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input,
                                    const Tensor& gamma, const Tensor& beta,
                                    const Tensor& running_mean, const Tensor& running_var,
                                    float momentum, float epsilon, bool training);
};

struct LayerNormOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input,
                                    const Tensor& gamma, const Tensor& beta, float epsilon);
};

struct GELUOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x);
};

struct ReshapeOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x, const std::vector<int64_t>& new_shape);
};

struct TransposeOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x, int64_t dim1, int64_t dim2);
};

struct MeanOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x, const std::vector<int64_t>& dims, bool keepdim = false);
};

class Autograd {
public:
    Graph& graph() { return graph_; }
    const Graph& graph() const { return graph_; }
    GradientMap& gradients() { return grads_; }
    const GradientMap& gradients() const { return grads_; }
    void clear_gradients() { grads_.clear(); }
    Expected<void> backward(Runtime& runtime, const Tensor& loss);

private:
    Graph graph_;
    GradientMap grads_;
};

} // namespace axon
