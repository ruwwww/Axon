# Polymorphic Node DAG Autograd Engine (PyTorch Autograd Model)

## Context

Axon's initial Phase 1 autograd design recorded backward graph operations as a flat list of passive `GraphNode` structs containing an `OpType` enum. During `Autograd::backward()`, a 15-case `switch(node.op)` statement dispatched backward logic to static operation functions (`MatMulOp::backward`, `ReLUOp::backward`, etc.).

As Axon scaled to support new compute backends (AVX2 SIMD, OpenCL iGPU) and custom user operations, this central switch model introduced several pain points:
1. Adding a new operation required modifying core autograd files (`autograd.h`, `autograd.cpp`).
2. Operation backward implementations were coupled to central dispatch logic rather than co-located with forward definitions.
3. Node attributes (such as kernel size, stride, or scale metadata) had to be packed into generic `Tensor op_data` buffers.

## Decision

We replaced the passive `GraphNode` vector, `OpType` enum, and central `switch` dispatch with a **Polymorphic `Node` DAG Autograd Engine** matching PyTorch's `torch::autograd::Node` model.

1. **Interface**: Abstract `Node` base class in `include/axon/autograd/node.h`:
   ```cpp
   class Node {
   public:
       virtual ~Node() = default;
       virtual Expected<void> apply(Runtime& runtime, GradientMap& grads) = 0;
       virtual std::vector<Tensor>& inputs() = 0;
       virtual const std::vector<Tensor>& inputs() const = 0;
       virtual std::string name() const { return "Node"; }
   };
   ```
2. **Concrete Nodes**: Each operation defines its own `Node` subclass (e.g. `MatMulNode`, `ReLUNode`, `Conv2DNode`) holding its own forward input references and operation parameters.
3. **Graph Storage**: `Graph` stores `std::vector<std::shared_ptr<Node>>`.
4. **Decoupled Execution**: `Autograd::backward()` iterates the node list in reverse and executes `node->apply(runtime, grads)` polymorphically without switch statements or enum checks.

## Consequences

- **Extensibility**: New operations can be added by implementing a `Node` subclass without editing `autograd.h` or `autograd.cpp`.
- **Encapsulation**: Node attributes (stride, padding, epsilon, etc.) are typed member variables rather than opaque metadata buffers.
- **Maintainability**: `OpType` enum and `struct GraphNode` are completely removed from the codebase.
