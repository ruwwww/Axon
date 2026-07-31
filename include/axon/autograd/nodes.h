#pragma once

#include "axon/autograd/node.h"

namespace axon {

class MatMulNode : public Node {
public:
    MatMulNode(Tensor a, Tensor b, Tensor output)
        : output_(std::move(output)), input_ids_{a.id(), b.id()}, inputs_{std::move(a), std::move(b)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "MatMul"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class ReLUNode : public Node {
public:
    ReLUNode(Tensor x, Tensor output)
        : output_(std::move(output)), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "ReLU"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class AddNode : public Node {
public:
    AddNode(Tensor a, Tensor b, Tensor output)
        : output_(std::move(output)), input_ids_{a.id(), b.id()}, inputs_{std::move(a), std::move(b)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "Add"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class SubNode : public Node {
public:
    SubNode(Tensor a, Tensor b, Tensor output)
        : output_(std::move(output)), input_ids_{a.id(), b.id()}, inputs_{std::move(a), std::move(b)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "Sub"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class MulScalarNode : public Node {
public:
    MulScalarNode(Tensor x, Tensor output, float scalar)
        : output_(std::move(output)), scalar_(scalar), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "MulScalar"; }

private:
    Tensor output_;
    float scalar_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class DivScalarNode : public Node {
public:
    DivScalarNode(Tensor x, Tensor output, float scalar)
        : output_(std::move(output)), scalar_(scalar), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "DivScalar"; }

private:
    Tensor output_;
    float scalar_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class Conv2DNode : public Node {
public:
    Conv2DNode(Tensor input, Tensor weight, Tensor bias, Tensor output, int64_t stride, int64_t padding)
        : output_(std::move(output)), stride_(stride), padding_(padding)
        , input_ids_{input.id(), weight.id(), bias.id()}
        , inputs_{std::move(input), std::move(weight), std::move(bias)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "Conv2D"; }

private:
    Tensor output_;
    int64_t stride_;
    int64_t padding_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class MaxPool2dNode : public Node {
public:
    MaxPool2dNode(Tensor input, Tensor output, int64_t kernel, int64_t stride)
        : output_(std::move(output)), kernel_(kernel), stride_(stride)
        , input_ids_{input.id()}, inputs_{std::move(input)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "MaxPool2d"; }

private:
    Tensor output_;
    int64_t kernel_;
    int64_t stride_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class AvgPool2dNode : public Node {
public:
    AvgPool2dNode(Tensor input, Tensor output, int64_t kernel, int64_t stride)
        : output_(std::move(output)), kernel_(kernel), stride_(stride)
        , input_ids_{input.id()}, inputs_{std::move(input)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "AvgPool2d"; }

private:
    Tensor output_;
    int64_t kernel_;
    int64_t stride_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class BatchNormNode : public Node {
public:
    BatchNormNode(Tensor input, Tensor gamma, Tensor beta, Tensor running_mean, Tensor running_var,
                  Tensor output, float momentum, float epsilon, bool training)
        : output_(std::move(output)), momentum_(momentum), epsilon_(epsilon), training_(training)
        , input_ids_{input.id(), gamma.id(), beta.id(), running_mean.id(), running_var.id()}
        , inputs_{std::move(input), std::move(gamma), std::move(beta), std::move(running_mean), std::move(running_var)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "BatchNorm"; }

private:
    Tensor output_;
    float momentum_;
    float epsilon_;
    bool training_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class LayerNormNode : public Node {
public:
    LayerNormNode(Tensor input, Tensor gamma, Tensor beta, Tensor output, float epsilon)
        : output_(std::move(output)), epsilon_(epsilon)
        , input_ids_{input.id(), gamma.id(), beta.id()}
        , inputs_{std::move(input), std::move(gamma), std::move(beta)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "LayerNorm"; }

private:
    Tensor output_;
    float epsilon_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class GELUNode : public Node {
public:
    GELUNode(Tensor x, Tensor output)
        : output_(std::move(output)), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "GELU"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class ReshapeNode : public Node {
public:
    ReshapeNode(Tensor x, Tensor output)
        : output_(std::move(output)), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "Reshape"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class TransposeNode : public Node {
public:
    TransposeNode(Tensor x, Tensor output, int64_t dim1, int64_t dim2)
        : output_(std::move(output)), dim1_(dim1), dim2_(dim2), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "Transpose"; }

private:
    Tensor output_;
    int64_t dim1_;
    int64_t dim2_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class MeanNode : public Node {
public:
    MeanNode(Tensor x, Tensor output, std::vector<int64_t> dims, bool keepdim)
        : output_(std::move(output)), dims_(std::move(dims)), keepdim_(keepdim), input_ids_{x.id()}, inputs_{std::move(x)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "Mean"; }

private:
    Tensor output_;
    std::vector<int64_t> dims_;
    bool keepdim_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class CrossEntropyLossNode : public Node {
public:
    CrossEntropyLossNode(Tensor logits, Tensor targets, Tensor output, Tensor log_softmax_out)
        : output_(std::move(output)), log_softmax_out_(std::move(log_softmax_out))
        , input_ids_{logits.id(), targets.id()}
        , inputs_{std::move(logits), std::move(targets)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "CrossEntropyLoss"; }

private:
    Tensor output_;
    Tensor log_softmax_out_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class MSELossNode : public Node {
public:
    MSELossNode(Tensor pred, Tensor target, Tensor output)
        : output_(std::move(output)), input_ids_{pred.id(), target.id()}, inputs_{std::move(pred), std::move(target)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "MSE"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

class L1LossNode : public Node {
public:
    L1LossNode(Tensor pred, Tensor target, Tensor output)
        : output_(std::move(output)), input_ids_{pred.id(), target.id()}, inputs_{std::move(pred), std::move(target)} {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    std::vector<Tensor>& inputs() override { return inputs_; }
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::vector<TensorId> input_ids() const override { return input_ids_; }
    std::string name() const override { return "L1Loss"; }

private:
    Tensor output_;
    std::vector<TensorId> input_ids_;
    std::vector<Tensor> inputs_;
};

} // namespace axon
