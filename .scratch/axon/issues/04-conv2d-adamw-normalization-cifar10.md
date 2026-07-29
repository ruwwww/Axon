# 04 — Conv2D + AdamW + Normalization + CIFAR10

## Context

Axon is a minimal deep learning framework. This ticket adds convolutional and normalization layers, AdamW optimizer, and a CIFAR10 training example on top of the existing Module/training infrastructure. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture. Tickets 01–03 must be complete before this starts.

## What to build

The layers needed for convolutional networks and a more advanced optimizer:

- **Conv2D Module**: `in_channels x out_channels x kernel_size` convolution. Supports stride, padding. Forward: im2col + matmul or a naive direct loop. Backward computes d_input and d_weight via convolution transpose.
- **BatchNorm Module**: Per-channel normalization with learnable affine parameters (gamma, beta). Tracks running mean/variance during training. eval() mode uses running stats.
- **LayerNorm Module**: Per-sample normalization over the last N dimensions. No running statistics. Uses learnable affine parameters.
- **Embedding Module**: Lookup table mapping indices to dense vectors. Forward: `table[input]`.
- **Dropout Module**: Randomly zeros elements during training with probability p. Scales by 1/(1-p). No-op during eval.
- **Flatten Module**: Reshapes input to 1D per sample (preserving batch dim).
- **Sequential Module**: Container that chains modules. `forward(x)` calls each module in order. `parameters()` returns the union of all child parameters.
- **Residual Module**: `forward(x) = F(x) + x`. Skip connection.
- **AdamW Optimizer**: Adam with decoupled weight decay. Stores first/second moment estimates. Allocates state eagerly.
- **CIFAR10 Dataset**: Pre-downloaded binary files (CIFAR-10 batch format). `get(i)` returns (32x32x3 image, label).
- **CIFAR10 training example**: `examples/cifar10.cpp` — trains a simple ConvNet (Conv2D → ReLU → Pool → Linear) on CIFAR10.
- **CPU backend additions**: `cpu::conv2d(out, input, weight, stride, padding)`, `cpu::maxpool2d(out, input, kernel, stride)`, `cpu::batchnorm(out, input, gamma, beta, running_mean, running_var)`, `cpu::layernorm(out, input, gamma, beta)`.

## Key interfaces

```cpp
class Conv2D : public Module {
    Parameter weight_;  // [out_ch, in_ch, kh, kw]
    Parameter bias_;    // [out_ch]
public:
    Conv2D(Runtime& rt, size_t in_ch, size_t out_ch, size_t kernel, 
           size_t stride = 1, size_t padding = 0);
    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
};

class BatchNorm : public Module {
    Parameter gamma_, beta_;  // [C]
    Tensor running_mean_, running_var_;
    float momentum_;
public:
    BatchNorm(Runtime& rt, size_t channels, float momentum = 0.9);
    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
};

class LayerNorm : public Module {
    Parameter gamma_, beta_;
public:
    LayerNorm(Runtime& rt, std::span<const int64_t> normalized_shape);
    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
};

class Sequential : public Module {
    std::vector<std::unique_ptr<Module>> modules_;
public:
    void add(std::unique_ptr<Module> module);
    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
    std::vector<Parameter*> parameters() override;
};

class AdamW : public Optimizer {
    struct State {
        Tensor m;  // first moment
        Tensor v;  // second moment
    };
    std::vector<State> state_;
    float lr_, beta1_, beta2_, eps_, weight_decay_;
public:
    AdamW(Runtime& rt, std::vector<Parameter*> params, float lr,
          float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8,
          float weight_decay = 0.01);
    Expected<void> step() override;
    void zero_grad() override;
};
```

## Acceptance criteria

- [ ] Conv2D::forward produces correct output shape and values for a known input/kernel.
- [ ] Conv2D::backward computes correct d_input and d_weight (finite differences match).
- [ ] BatchNorm forward and backward produce correct results in train and eval modes.
- [ ] LayerNorm forward and backward produce correct results.
- [ ] Sequential chains modules correctly, parameters() returns union of all children.
- [ ] AdamW::step applies the correct update formula for a single Parameter.
- [ ] CIFAR10 dataset reads binary batch files correctly.
- [ ] CIFAR10 training example compiles, runs, and loss decreases over epochs.

## Blocked by

03 — Modules + SGD + Loss + DataLoader + MNIST

## Status

ready-for-agent
