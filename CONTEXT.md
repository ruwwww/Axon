# Axon

A minimal deep learning framework focused on understanding how modern AI runtimes are built.

## Language

**Tensor**:
A lightweight, copyable frontend object. Holds a `std::shared_ptr<Storage>`, an `int64_t storage_offset_` (in elements), and an immutable `TensorType` descriptor. Phase 1 uses shared ownership via shared_ptr for simplicity; the original spec's "non-owning pointer" model can be restored later if the refcount overhead matters.
_Avoid_: Raw owning pointers to storage

**TensorIterator**:
A strided accessor over a Tensor's data. Reads shape+strides+dtype+offset from a TensorType and Storage, and provides `operator[]` that computes the correct memory offset for any flat index via stride arithmetic. Has an internal `is_contiguous()` fast-path branch. Does not know about Runtime, backend kernels, or autograd. Lives in `include/axon/tensor/tensor_iterator.h`.
_Avoid_: Calling `data<T>()` on a potentially non-contiguous tensor and iterating with flat indices

**TensorId**:
A runtime-unique identifier assigned to each Tensor instance. Used for graph bookkeeping and debugging. Not a globally unique ID across processes or saved models.

**TensorType**:
An immutable descriptor bundling shape, stride, dtype, device, and quantization.

**Storage**:
The memory owner. A reference-counted block of bytes holding the raw tensor data. Owned by a higher-level manager (e.g. `StorageManager` or `shared_ptr<Storage>`); never by Tensor itself.

**QuantizationDescriptor**:
A description of the low-precision encoding scheme applied to a Storage block (block size, packing, scales, decode rule). Belongs to Storage, not Tensor.

**Graph**:
A linear sequence of operation nodes (`std::shared_ptr<Node>`), built during forward execution, consumed by autograd's backward pass. No optimization, no scheduling, no compiler.

**Node**:
Abstract base class for autograd graph nodes (`std::shared_ptr<Node>`). Created by an operation's `forward()` when `requires_grad` is enabled and appended to the Graph. Each operation defines a concrete `Node` subclass (e.g., `MatMulNode`, `ReLUNode`) holding saved forward tensors and implementing `apply(runtime, grads)` for backward gradient computation.

**Operation**:
A stateless functor implementing `forward()` and `backward()`. `forward()` allocates outputs, calls the backend kernel, and (if `requires_grad`) creates a concrete `Node` and appends it to the Graph. Operations own graph recording, not the Runtime.

**Autograd**:
Owns the Graph and GradientMap. Drives backward traversal — creates initial gradient, iterates nodes in reverse, calls each node's `apply(runtime, grads)`, and accumulates results in the GradientMap. After traversal, populates each tensor's `.grad`.

**GradientMap**:
Working storage during backward pass. Maps TensorId → gradient Tensor. Accumulates gradients in-place. Gradients are copied to Tensor::grad only after backward completes.

**Backend**:
The numerical kernel provider (e.g. CPU). Receives raw `Tensor*` pointers. Never accesses the Graph. Designed so a GPU backend can be added without changing Tensor's interface.

**Runtime**:
The execution context (allocator, graph reference, training mode). Owns the Allocator and the Graph reference. Serves as the public execution API — users call `runtime.matmul(...)` which delegates to `MatMulOp::forward()`. Not an executor: operations call into the backend and record graph nodes themselves. Minimal: no execution plan, no VM, no instruction set.

**Parameter**:
Wraps a Tensor to represent a trainable model parameter. Contains the Tensor, gradient, and trainable flag.
_Avoid_: Direct Tensor for trainable weights

**Module**:
Base class for neural network building blocks. Provides `forward()`, `parameters()`, `train()`, `eval()`. Parameters must be explicitly registered via `register_parameter()`.
_Avoid_: Automatic parameter discovery

**Optimizer**:
Owns optimizer state (momentum buffers, Adam moments). Receives `Runtime&` and `vector<Parameter*>` on construction. Allocates state tensors eagerly. Mutates parameters in-place via `step()`. `zero_grad()` zeros each parameter's gradient directly.

**Loss**:
An Operation subclass that computes a scalar error. Fuses numerical transforms (e.g. log-softmax + NLL in CrossEntropy) for stability. Has `forward()` and `backward()` like any other Operation. Not a Module.

**Dataset**:
An interface with `size()` and `get(index)`. Phase 1 is pre-downloaded files only, no caching or download logic.

**DataLoader**:
Synchronous, single-threaded. Iterates indices, calls `dataset.get(i)`, collates into a batch Tensor. No prefetch or async workers in Phase 1.

**Allocator**:
Owned by Runtime. Creates Storage blocks. Operations call `runtime.allocator().allocate(type)` to allocate output tensors. Not a global singleton — no global state.
