# 02 — Operations + Graph + Autograd

## Context

Axon is a minimal deep learning framework. This ticket builds the eager autograd system on top of the Storage/Tensor foundation. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture. Ticket 01 must be complete before this starts.

## What to build

The operation abstraction, graph recording, and autograd backward pass:

- **Operation pattern**: Stateless functor structs implementing `forward()` and `backward()`. Each operation is a struct with two static methods.
- **MatMulOp**: `forward(Runtime&, const Tensor& a, const Tensor& b)` — validates shapes, allocates output via Runtime allocator, calls `cpu::matmul(out, a, b)`, appends GraphNode if gradients are needed. `backward(Runtime&, const GraphNode&, GradientMap&)` — computes `da = grad_out @ b^T`, `db = a^T @ grad_out`, accumulates into GradientMap.
- **ReLUOp**: `forward(Runtime&, const Tensor& x)` — calls `cpu::relu(out, x)`, records mask for backward. `backward` — computes `dx = grad_out * (x > 0)`.
- **GraphNode**: Stores the operation pointer/enum, input tensors, output tensor, and a non-owning `Runtime*` pointer (set during forward, used by backward for allocation).
- **Graph**: A linear `std::vector<GraphNode>`. Append during forward, iterate in reverse during backward. No optimization, no scheduling.
- **Autograd**: `backward(runtime, graph, grads)` — takes the output tensor, creates initial gradient of 1.0 (for scalar loss), iterates graph nodes in reverse calling each op's backward, accumulates into GradientMap. After traversal, copies gradients to `Tensor::grad`.
- **GradientMap**: `std::unordered_map<TensorId, Tensor>`. Accumulates gradients in-place during backward.
- **Runtime methods**: `Runtime::matmul(a, b)`, `Runtime::relu(x)` — delegate to the respective Operation's forward.
- **CPU backend additions**: `cpu::matmul(out, a, b)`, `cpu::relu(out, x)` — basic naive implementations (no BLAS, just loops for Phase 1).

## Key interfaces

```cpp
// GraphNode
struct GraphNode {
    OpType op;                      // enum: MatMul, ReLU, ...
    std::vector<Tensor> inputs;
    Tensor output;
    Runtime* runtime;               // non-owning, for backward allocation
    // Optional: per-op data (e.g., ReLU mask — the input x for mask computation)
    Tensor op_data;
};

// Graph
class Graph {
    std::vector<GraphNode> nodes_;
public:
    void append(GraphNode node);
    size_t size() const;
    const GraphNode& operator[](size_t i) const;
};

// Gradients
using GradientMap = std::unordered_map<TensorId, Tensor>;

// Autograd
class Autograd {
    Graph graph_;
public:
    Graph& graph() { return graph_; }
    Expected<void> backward(Runtime& runtime, const Tensor& loss);
};

// Operations
struct MatMulOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& a, const Tensor& b);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};
struct ReLUOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& x);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

// CPU backend additions
namespace cpu {
    Expected<void> matmul(Tensor& out, const Tensor& a, const Tensor& b);
    Expected<void> relu(Tensor& out, const Tensor& x);
}

// Autograd is owned by Runtime
class Runtime {
    Autograd autograd_;
public:
    Autograd& autograd() { return autograd_; }
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& x);
};
```

## Acceptance criteria

- [ ] `MatMulOp::forward` allocates correct output shape and delegates to `cpu::matmul`.
- [ ] `ReLUOp::forward` allocates correct output shape and delegates to `cpu::relu`.
- [ ] With `requires_grad=true`, forward appends a GraphNode to the graph.
- [ ] With `requires_grad=false`, no GraphNode is recorded.
- [ ] Graph nodes are recorded in order and iterated in reverse during backward.
- [ ] `c.backward()` on a scalar loss produces a GradientMap entry for each input tensor.
- [ ] Autograd integration test: `z = relu(matmul(W, x))`, finite differences match analytical gradients.
- [ ] Gradient accumulation works: if a tensor is used in multiple ops, gradients add.
- [ ] GradientMap is cleared after each backward pass (or on request).

## Testing decisions

- Operation tests verify the right backend function is called with the right shapes. Do not re-prove kernel math.
- Autograd tests verify finite differences match on small graphs. This is the gate.

## Blocked by

01 — Foundation: Storage + Tensor + Allocator + CPU Backend

## Status

ready-for-agent
