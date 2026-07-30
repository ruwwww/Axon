#pragma once

#include "axon/autograd/node.h"

namespace axon {

class MatMulNode : public Node {
public:
    MatMulNode(Tensor a, Tensor b, Tensor output)
        : a_(std::move(a)), b_(std::move(b)), output_(std::move(output)), inputs_({a_, b_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "MatMul"; }

private:
    Tensor a_;
    Tensor b_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

class ReLUNode : public Node {
public:
    ReLUNode(Tensor x, Tensor output)
        : x_(std::move(x)), output_(std::move(output)), inputs_({x_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "ReLU"; }

private:
    Tensor x_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

class AddNode : public Node {
public:
    AddNode(Tensor a, Tensor b, Tensor output)
        : a_(std::move(a)), b_(std::move(b)), output_(std::move(output)), inputs_({a_, b_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "Add"; }

private:
    Tensor a_;
    Tensor b_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

class Conv2DNode : public Node {
public:
    Conv2DNode(Tensor input, Tensor weight, Tensor bias, Tensor output, int64_t stride, int64_t padding)
        : input_(std::move(input)), weight_(std::move(weight)), bias_(std::move(bias))
        , output_(std::move(output)), stride_(stride), padding_(padding)
        , inputs_({input_, weight_, bias_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "Conv2D"; }

private:
    Tensor input_;
    Tensor weight_;
    Tensor bias_;
    Tensor output_;
    int64_t stride_;
    int64_t padding_;
    std::vector<Tensor> inputs_;
};

class MaxPool2dNode : public Node {
public:
    MaxPool2dNode(Tensor input, Tensor output, int64_t kernel, int64_t stride)
        : input_(std::move(input)), output_(std::move(output))
        , kernel_(kernel), stride_(stride), inputs_({input_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "MaxPool2d"; }

private:
    Tensor input_;
    Tensor output_;
    int64_t kernel_;
    int64_t stride_;
    std::vector<Tensor> inputs_;
};

class AvgPool2dNode : public Node {
public:
    AvgPool2dNode(Tensor input, Tensor output, int64_t kernel, int64_t stride)
        : input_(std::move(input)), output_(std::move(output))
        , kernel_(kernel), stride_(stride), inputs_({input_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "AvgPool2d"; }

private:
    Tensor input_;
    Tensor output_;
    int64_t kernel_;
    int64_t stride_;
    std::vector<Tensor> inputs_;
};

class BatchNormNode : public Node {
public:
    BatchNormNode(Tensor input, Tensor gamma, Tensor beta, Tensor running_mean, Tensor running_var,
                  Tensor output, float momentum, float epsilon, bool training)
        : input_(std::move(input)), gamma_(std::move(gamma)), beta_(std::move(beta))
        , running_mean_(std::move(running_mean)), running_var_(std::move(running_var))
        , output_(std::move(output)), momentum_(momentum), epsilon_(epsilon), training_(training)
        , inputs_({input_, gamma_, beta_, running_mean_, running_var_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "BatchNorm"; }

private:
    Tensor input_;
    Tensor gamma_;
    Tensor beta_;
    Tensor running_mean_;
    Tensor running_var_;
    Tensor output_;
    float momentum_;
    float epsilon_;
    bool training_;
    std::vector<Tensor> inputs_;
};

class LayerNormNode : public Node {
public:
    LayerNormNode(Tensor input, Tensor gamma, Tensor beta, Tensor output, float epsilon)
        : input_(std::move(input)), gamma_(std::move(gamma)), beta_(std::move(beta))
        , output_(std::move(output)), epsilon_(epsilon)
        , inputs_({input_, gamma_, beta_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "LayerNorm"; }

private:
    Tensor input_;
    Tensor gamma_;
    Tensor beta_;
    Tensor output_;
    float epsilon_;
    std::vector<Tensor> inputs_;
};

class GELUNode : public Node {
public:
    GELUNode(Tensor x, Tensor output)
        : x_(std::move(x)), output_(std::move(output)), inputs_({x_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "GELU"; }

private:
    Tensor x_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

class ReshapeNode : public Node {
public:
    ReshapeNode(Tensor x, Tensor output)
        : x_(std::move(x)), output_(std::move(output)), inputs_({x_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "Reshape"; }

private:
    Tensor x_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

class TransposeNode : public Node {
public:
    TransposeNode(Tensor x, Tensor output, int64_t dim1, int64_t dim2)
        : x_(std::move(x)), output_(std::move(output)), dim1_(dim1), dim2_(dim2), inputs_({x_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "Transpose"; }

private:
    Tensor x_;
    Tensor output_;
    int64_t dim1_;
    int64_t dim2_;
    std::vector<Tensor> inputs_;
};

class MeanNode : public Node {
public:
    MeanNode(Tensor x, Tensor output, std::vector<int64_t> dims, bool keepdim)
        : x_(std::move(x)), output_(std::move(output)), dims_(std::move(dims)), keepdim_(keepdim), inputs_({x_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "Mean"; }

private:
    Tensor x_;
    Tensor output_;
    std::vector<int64_t> dims_;
    bool keepdim_;
    std::vector<Tensor> inputs_;
};

class CrossEntropyLossNode : public Node {
public:
    CrossEntropyLossNode(Tensor logits, Tensor targets, Tensor output, Tensor log_softmax_out)
        : logits_(std::move(logits)), targets_(std::move(targets))
        , output_(std::move(output)), log_softmax_out_(std::move(log_softmax_out))
        , inputs_({logits_, targets_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "CrossEntropyLoss"; }

private:
    Tensor logits_;
    Tensor targets_;
    Tensor output_;
    Tensor log_softmax_out_;
    std::vector<Tensor> inputs_;
};

class MSELossNode : public Node {
public:
    MSELossNode(Tensor pred, Tensor target, Tensor output)
        : pred_(std::move(pred)), target_(std::move(target)), output_(std::move(output)), inputs_({pred_, target_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "MSE"; }

private:
    Tensor pred_;
    Tensor target_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

class L1LossNode : public Node {
public:
    L1LossNode(Tensor pred, Tensor target, Tensor output)
        : pred_(std::move(pred)), target_(std::move(target)), output_(std::move(output)), inputs_({pred_, target_}) {}

    Expected<void> apply(Runtime& runtime, GradientMap& grads) override;
    const std::vector<Tensor>& inputs() const override { return inputs_; }
    std::string name() const override { return "L1Loss"; }

private:
    Tensor pred_;
    Tensor target_;
    Tensor output_;
    std::vector<Tensor> inputs_;
};

} // namespace axon
