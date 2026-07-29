#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime;

enum class OpType : uint8_t {
    MatMul,
    ReLU,
    Add,
    CrossEntropyLoss,
    MSE,
    Conv2D,
    MaxPool2d,
    AvgPool2d,
    BatchNorm,
    LayerNorm,
    GELU,
};

struct GraphNode {
    OpType op;
    std::vector<Tensor> inputs;
    Tensor output;
    Runtime* runtime;
    Tensor op_data;
};

class Graph {
public:
    void append(GraphNode node);
    void clear() { nodes_.clear(); }
    size_t size() const;
    const GraphNode& operator[](size_t i) const;
    GraphNode& operator[](size_t i);

private:
    std::vector<GraphNode> nodes_;
};

using GradientMap = std::unordered_map<TensorId, Tensor>;

struct MatMulOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& a, const Tensor& b);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct ReLUOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct AddOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& a, const Tensor& b);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct Conv2DOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input, const Tensor& weight,
                                    const Tensor& bias, int64_t stride, int64_t padding);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct MaxPool2dOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input, int64_t kernel, int64_t stride);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct AvgPool2dOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input, int64_t kernel, int64_t stride);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct BatchNormOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input,
                                    const Tensor& gamma, const Tensor& beta,
                                    const Tensor& running_mean, const Tensor& running_var,
                                    float momentum, float epsilon, bool training);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct LayerNormOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& input,
                                    const Tensor& gamma, const Tensor& beta, float epsilon);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

struct GELUOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
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
