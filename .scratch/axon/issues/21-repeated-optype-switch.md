# 21 — Polymorphic Node DAG Autograd Engine (PyTorch Autograd Model)

## What to build

Replace the passive `GraphNode` vector, `OpType` enum, and the 15-case `switch(node.op)` in `Autograd::backward()` with a **Polymorphic `Node` DAG Autograd Engine** matching PyTorch's `torch::autograd::Node` model.

Introduce an abstract `Node` base class in `include/axon/autograd/node.h`:
```cpp
class Node {
public:
    virtual ~Node() = default;
    virtual Expected<void> apply(Runtime& runtime, GradientMap& grads) = 0;
};
```

Each operation defines its own concrete `Node` subclass (e.g. `MatMulNode`, `ReLUNode`, `Conv2DNode`) holding its forward inputs/saved tensors. During forward pass, operations instantiate their concrete `Node` and push `std::shared_ptr<Node>` into the `Graph` DAG.

`Autograd::backward()` iterates the `Graph` nodes in reverse and calls `node->apply(runtime, grads)`. This completely eliminates the `OpType` enum, `GraphNode` struct, and central `switch` statements across the codebase.

## Blocked by

None — can start immediately. Priority: P0 (Prerequisite for AVX2 and OpenCL backends).

## Acceptance criteria

- [ ] Abstract `Node` base class with virtual `apply(Runtime&, GradientMap&)` method created in `include/axon/autograd/node.h`
- [ ] Passive `GraphNode` struct and `OpType` enum completely eliminated from the codebase
- [ ] Each operation (`MatMul`, `ReLU`, `Add`, `Conv2D`, `MaxPool2d`, `AvgPool2d`, `BatchNorm`, `LayerNorm`, `GELU`, `Reshape`, `Transpose`, `Mean`, `CrossEntropyLoss`, `MSELoss`, `L1Loss`) defines its own `Node` subclass
- [ ] `Graph` stores `std::vector<std::shared_ptr<Node>>`
- [ ] `Autograd::backward()` traverses `Node` graph and executes `node->apply(runtime, grads)` without enum or switch dispatch
- [ ] All 190 existing test cases in `axon_tests.exe` pass unchanged
- [ ] Adding a new operation requires zero edits to autograd core files (`autograd.h` or `autograd.cpp`)

## Status

ready-for-agent
