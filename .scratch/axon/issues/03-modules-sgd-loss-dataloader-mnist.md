# 03 — Modules + SGD + Loss + DataLoader + MNIST

## Context

Axon is a minimal deep learning framework. This ticket builds the training loop infrastructure on top of the autograd system. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture. Tickets 01 and 02 must be complete before this starts.

## What to build

The neural network building blocks and the first end-to-end training example:

- **Parameter**: Wraps a Tensor with a separate gradient Tensor and a `trainable` flag. Gradient is allocated lazily on first backward.
- **Module**: Base class for all layers. Provides `register_parameter(name, &param)`, `parameters()` returning a `std::vector<Parameter*>`, `train()`, `eval()`.
- **Linear Module**: `in_features x out_features` weight matrix + optional bias. `forward(x)` computes `y = x @ W^T + b`. Registers weight and bias as parameters.
- **CrossEntropyLossOp**: Operation that fuses log-softmax + NLL loss for numerical stability. Forward: logits → log-softmax → NLL → scalar loss. Backward computes gradient analytically.
- **MSELossOp**: Operation computing mean squared error. Forward: `mean((pred - target)^2)`. Backward: `d_pred = 2 * (pred - target) / N`.
- **SGD Optimizer**: Constructor takes `Runtime&` and `vector<Parameter*>`, plus learning rate and momentum. `step()` applies SGD update: `param = param - lr * grad + momentum * prev_update`. `zero_grad()` zeros each parameter's gradient. Allocates momentum buffers eagerly.
- **Dataset interface**: `size_t size()` and `Tensor get(size_t index)`. 
- **MNIST Dataset**: Pre-downloaded files only. Reads IDX3-UBIT (images) and IDX1-UBIT (labels) format files. `get(i)` returns a pair of Tensors (image, label).
- **DataLoader**: Synchronous, single-threaded. Constructor takes Dataset&, batch_size, shuffle flag. Iterates shuffled indices, calls `dataset.get(i)` for each, collates into a batch Tensor. Returns batches on iteration.
- **MNIST training example**: `examples/mnist.cpp` — a complete training script that loads MNIST, creates a Linear model (784→10), trains with SGD for N epochs, reports loss.

## Key interfaces

```cpp
// Parameter
class Parameter {
    Tensor tensor_;
    Tensor grad_;    // lazy allocation
    bool trainable_;
public:
    Tensor& tensor() { return tensor_; }
    Tensor& grad() { return grad_; }
    bool trainable() const { return trainable_; }
};

// Module
class Module {
    std::vector<Parameter*> parameters_;
public:
    void register_parameter(const std::string& name, Parameter* param);
    std::vector<Parameter*> parameters();
    virtual Expected<Tensor> forward(Runtime& rt, const Tensor& x) = 0;
    void train();  // sets training mode
    void eval();   // sets evaluation mode
};

// Linear
class Linear : public Module {
    Parameter weight_;
    Parameter bias_;   // optional
public:
    Linear(Runtime& rt, size_t in, size_t out, bool bias = true);
    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
};

// CrossEntropyLossOp
struct CrossEntropyLossOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& logits, const Tensor& target);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

// MSELossOp
struct MSELossOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& pred, const Tensor& target);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

// SGD
class SGD {
    Runtime& rt_;
    std::vector<Parameter*> params_;
    float lr_;
    float momentum_;
    std::vector<Tensor> momentum_bufs_;  // state
public:
    SGD(Runtime& rt, std::vector<Parameter*> params, float lr, float momentum = 0.0);
    Expected<void> step();
    void zero_grad();
};

// Dataset
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> get(size_t index) = 0;  // (input, target)
};

// MNIST
class MNIST : public Dataset {
public:
    MNIST(Runtime& rt, const std::string& path, bool train);  // pre-downloaded files
    size_t size() const override;
    std::pair<Tensor, Tensor> get(size_t index) override;
};

// DataLoader
class DataLoader {
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
public:
    DataLoader(Dataset& dataset, size_t batch_size, bool shuffle = true);
    struct Batch { Tensor inputs; Tensor targets; };
    std::vector<Batch> iter();  // or iterator interface
};
```

## Acceptance criteria

- [ ] Linear::forward produces correct shapes and values for a random input.
- [ ] Parameters are correctly registered and returned by Module::parameters().
- [ ] CrossEntropyLossOp computes correct loss and gradient (finite differences match).
- [ ] SGD::step applies the correct update formula on a single Parameter with known gradient.
- [ ] SGD::zero_grad zeros all parameter gradients.
- [ ] MNIST dataset reads IDX files correctly and returns correct shapes.
- [ ] DataLoader produces batches of the correct shape.
- [ ] MNIST training example compiles, runs, and loss decreases over epochs.
- [ ] Optimizer unit test: single Parameter with known grad, step(), assert new value matches hand calculation.

## Blocked by

01 — Foundation: Storage + Tensor + Allocator + CPU Backend
02 — Operations + Graph + Autograd

## Status

ready-for-agent
