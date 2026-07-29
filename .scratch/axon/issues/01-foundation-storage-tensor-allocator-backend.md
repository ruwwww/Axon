# 01 — Foundation: Storage + Tensor + Allocator + CPU Backend

## Context

Axon is a minimal deep learning framework. This ticket builds the memory and computation foundation that every other subsystem depends on. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture.

## What to build

The core data structures and the simplest CPU kernels:

- **Storage**: A reference-counted (`shared_ptr`) block of raw bytes. Holds `void* data`, `size_t size_bytes`, `size_t alignment`, and an optional `QuantizationDescriptor`. Quantization formats are defined as an enum (None, Q8_0, Q6_K, Q5_K, Q4_0, Q4_K, Q3_K, Q2_K) but only `None` is implemented in this ticket.
- **TensorType**: An immutable descriptor bundling shape (vector of int64_t), stride (vector of int64_t), dtype (enum: Float32, Float64, Int32, Int8, etc.), device (enum: CPU), and quantization.
- **Tensor**: A lightweight frontend holding `TensorId` (uint64_t), `TensorType`, `shared_ptr<Storage>`, and `bool requires_grad`. Copyable — copying shares the same Storage (refcount bump). Methods: `Tensor::zeros(runtime, shape)`, `Tensor::ones(runtime, shape)`, `Tensor::randn(runtime, shape)`, `Tensor::empty(runtime, shape)`.
- **Allocator**: Owned by Runtime. `allocate(TensorType)` returns a `shared_ptr<Storage>` of the right size. Alignment is `alignof(max_align_t)` for Phase 1.
- **Runtime skeleton**: Minimal class owning an `Allocator`. Provides `zeros()`, `ones()`, `randn()`, `empty()` factory methods. No autograd, no graph yet.
- **CPU Backend**: A namespace `cpu::` of free functions. Phase 1 kernels: `add`, `sub`, `mul`, `div` — all operating on raw `Tensor*` pointers. Each validates shapes match, returns `Expected<void>` on error.
- **Expected<T>**: A simple result type (`axon/core/expected.h`) — either holds a value of type T, or an error. Used as the return type for all cross-subsystem public APIs.

## Key interfaces

```cpp
// Storage
struct Storage {
    void* data;
    size_t size_bytes;
    size_t alignment;
    QuantizationDescriptor quant;
};

// TensorType
struct TensorType {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DType dtype;
    Device device;
    Quantization quant;
};

// Tensor
class Tensor {
    TensorId id;
    TensorType type_;
    std::shared_ptr<Storage> storage_;
    bool requires_grad_ = false;
public:
    static Tensor zeros(Runtime& rt, std::span<const int64_t> shape, DType dtype = Float32);
    static Tensor ones(Runtime& rt, std::span<const int64_t> shape, DType dtype = Float32);
    static Tensor randn(Runtime& rt, std::span<const int64_t> shape, DType dtype = Float32);
    static Tensor empty(Runtime& rt, std::span<const int64_t> shape, DType dtype = Float32);
};

// Allocator
class Allocator {
public:
    std::shared_ptr<Storage> allocate(const TensorType& type);
};

// Runtime (skeleton)
class Runtime {
    Allocator allocator_;
public:
    Allocator& allocator() { return allocator_; }
    Tensor zeros(std::span<const int64_t> shape, DType dtype = Float32);
    Tensor ones(std::span<const int64_t> shape, DType dtype = Float32);
    Tensor randn(std::span<const int64_t> shape, DType dtype = Float32);
    Tensor empty(std::span<const int64_t> shape, DType dtype = Float32);
};

// CPU Backend
namespace cpu {
    Expected<void> add(Tensor& out, const Tensor& a, const Tensor& b);
    Expected<void> sub(Tensor& out, const Tensor& a, const Tensor& b);
    Expected<void> mul(Tensor& out, const Tensor& a, const Tensor& b);
    Expected<void> div(Tensor& out, const Tensor& a, const Tensor& b);
}
```

## Acceptance criteria

- [ ] Storage can be created via Allocator, holds the requested byte count.
- [ ] Tensor::zeros creates a tensor filled with zeros, Tensor::ones filled with ones, Tensor::randn filled with random normal values.
- [ ] Copying a Tensor shares the same Storage (refcount increments).
- [ ] Tensor views share Storage with the parent.
- [ ] CPU backend element-wise ops (add, sub, mul, div) produce correct numerical results.
- [ ] Expected<T> correctly propagates errors from mismatched shape operations.
- [ ] Build system: CMake 3.22+, Catch2 v3 tests, C++20.

## Blocked by

None — can start immediately.

## Status

ready-for-agent
