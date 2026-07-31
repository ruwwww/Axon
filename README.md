<p align="center">
  <h1 align="center">Axon</h1>
  <p align="center">
    <strong>A minimal deep learning framework in C++20</strong>
  </p>
  <p align="center">
    <em>Eager autograd, AVX2 SIMD CPU backend, polymorphic Node DAG, GGML-style quantization — built from scratch.</em>
  </p>
  <p align="center">
    <a href="#features">Features</a> · <a href="#quick-start">Quick Start</a> · <a href="#architecture">Architecture</a> · <a href="#api-reference">API Reference</a> · <a href="#examples">Examples</a>
  </p>
</p>

---

## Overview

Axon is a minimal deep learning framework in C++20 that implements a complete training pipeline — tensors, autograd, neural network modules, optimizers, data loading, serialization, and quantization — with **zero external ML framework dependencies**.

Designed to show how modern AI runtimes are built: eager execution, polymorphic `Node` autograd recording, generic `KernelRegistry` dispatching, AVX2 SIMD CPU vectorization, and quantized inference formats inspired by GGML.

The framework is capable enough to train a ResNet-18 on CIFAR-10 and run quantized INT4 inference.

## Features

- **Polymorphic Node DAG Autograd** — PyTorch-style polymorphic `Node` DAG execution; operations record forward dependencies and backward replays in reverse with gradient accumulation
- **AVX2 SIMD CPU Backend** — Generic `KernelContext` & `KernelRegistry` dispatcher, runtime CPUID/OSXSAVE hardware query, 256-bit SIMD `Vec<T, ISA>` traits, vectorized element-wise ops, tiled FP32 GEMM (3.20x benchmark speedup), and quantized dot-products
- **Tensor Engine** — Multi-dimensional arrays with shared ownership (`shared_ptr<Storage>`), strided access, non-contiguous views, and dtype/device/quantization metadata
- **Neural Network Modules** — Linear, Conv2D, BatchNorm, LayerNorm, Dropout, Flatten, Embedding, Residual, Sequential
- **ResNet-18** — Complete implementation with BasicBlock and configurable output classes
- **Optimizers** — SGD (with momentum) and AdamW
- **Loss Functions** — Cross-entropy (fused log-softmax + NLL), MSE, and L1 loss
- **Data Pipeline** — CIFAR-10 and MNIST loaders with batching and shuffling
- **GGML-style Quantization** — Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K block-wise quantization for inference
- **Serialization** — Save/load individual tensors and full model checkpoints
- **Error Handling** — `Expected<T>` return types across all subsystem boundaries (no exceptions)
- **Zero ML Dependencies** — Only C++20 standard library and Catch2 v3 (tests)

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| C++ Compiler | C++20 (GCC 12+, Clang 14+, MSVC 19.30+) |
| CMake | 3.22+ |

### Build

```bash
git clone https://github.com/ruwwww/Axon
cd Axon
cmake -B build
cmake --build build
```

This produces:

| Target | Description |
|---|---|
| `axon` | Static library |
| `axon_tests` | Unit test suite (Catch2 v3) |
| `mnist_example` | MNIST training example |
| `cifar10_example` | CIFAR-10 training example |

### Run Tests

```bash
cd build && ./axon_tests
```

All **193 tests pass** (2401 assertions).

### Train on MNIST

Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) into `datasets/`:

```
datasets/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

```bash
./build/mnist_example datasets/
```

### Train on CIFAR-10

Download the [CIFAR-10 binary dataset](https://www.cs.toronto.edu/~kriz/cifar.html) into `cifar10_real_data/`:

```
cifar10_real_data/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
└── test_batch.bin
```

```bash
./build/cifar10_example cifar10_real_data/
```

## Architecture

### Project Structure

```
Axon/
├── include/axon/
│   ├── autograd/           # Graph, Node base class, concrete Node subclasses, GradientMap
│   ├── backend/            # cpuid.h, registry.h, cpu_backend.h, simd/ (vec.h, scalar.h, avx2.h)
│   ├── core/               # Expected<T>, types (DType, Device, QuantFormat), serialization
│   ├── data/               # Dataset interface, DataLoader, CIFAR10, MNIST
│   ├── nn/                 # Module base, Parameter, Sequential, Linear, Conv2D, BatchNorm,
│   │                       # LayerNorm, Dropout, Flatten, Embedding, Residual, ResNet18,
│   │                       # SGD, AdamW, CrossEntropyLossOp, MSELossOp, L1LossOp
│   ├── runtime/            # Runtime (public API), Allocator
│   ├── storage/            # Storage (aligned memory), QuantizationDescriptor
│   └── tensor/             # Tensor, TensorType, TensorIterator
├── src/                    # Implementation files (src/backend/cpu/ scalar/ & avx2/ translation units)
├── tests/                  # Catch2 v3 test suite (28 files, 193 tests)
├── examples/
│   ├── cifar10.cpp         # ConvNet training on CIFAR-10
│   ├── mnist.cpp           # Linear classifier on MNIST
│   └── mnist_benchmark_minimal.cpp
├── docs/
│   ├── adr/                # Architecture Decision Records (0001 through 0006)
│   ├── quantization_benchmark_formal.md
│   └── mnist_benchmark_blog.md
├── SPEC.md                 # Full project specification
├── CONTEXT.md              # Domain glossary
└── CMakeLists.txt
```

### Design

```
                    ┌──────────────┐
                    │   Runtime    │  Public API — owns Allocator + Autograd
                    │  (execution  │  Users call runtime.matmul(...), runtime.relu(...)
                    │   context)   │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐  ┌──────────────┐  ┌──────────────┐
   │  Allocator │  │  Kernel-     │  │  Autograd    │
   │ (creates   │  │  Registry    │  │ (owns Graph  │
   │  Storage)  │  │ (Scalar/AVX2 │  │  + Gradient- │
   └────────────┘  │  dispatch)   │  │  Map)        │
                   └──────────────┘  └──────┬───────┘
                                            │
                                     ┌──────┴──────┐
                                     │    Graph    │
                                     │   Node[]    │
                                     │ Polymorphic │
                                     │ Node DAG    │
                                     └─────────────┘
```

### Key design decisions

- **Runtime is the public API** — Operations call into the generic `KernelRegistry` dispatcher and record polymorphic `Node` DAG nodes themselves.
- **Operations own graph recording** — each operation's `forward()` checks `requires_grad`, allocates output, calls the backend kernel dispatcher, and appends a concrete `Node` (`shared_ptr<Node>`) to the `Graph` if needed.
- **Generic KernelRegistry & KernelContext** — Kernels conform to `KernelContext` (`outputs`, `inputs`, `attributes`) and are looked up by `(OpName, ISA)`.
- **Per-Translation-Unit AVX2 Isolation** — `src/backend/cpu/avx2/*.cpp` files compile with `/arch:AVX2` or `-mavx2 -mfma` in `CMakeLists.txt`, preserving unflagged binary portability.
- **No exceptions** — all fallible operations return `Expected<T>`, a `std::variant<T, Error>`.
- **No global state** — Allocator is owned by Runtime.

### Domain terminology

| Term | Meaning |
|---|---|
| Tensor | Lightweight copyable handle to n-dimensional data |
| Storage | Reference-counted aligned memory block |
| TensorType | Immutable shape/strides/dtype/device/quant descriptor |
| Runtime | Execution context (allocator, autograd, training mode) |
| KernelContext | Universal execution boundary containing output/input spans & attributes |
| KernelRegistry | Central dispatcher mapping `(OpName, ISA)` to execution kernel pointers |
| Vec\<T, ISA\> | 256-bit SIMD trait abstraction wrapping vector intrinsics |
| Graph | Vector of polymorphic `Node` objects recorded during forward pass |
| Node | Abstract base class for autograd nodes; implements virtual `apply()` for backward traversal |
| Autograd | Owns Graph + GradientMap; drives backward traversal |
| Module | Base class for neural network building blocks |
| Parameter | Wraps a Tensor with gradient and trainable flag |
| Expected\<T\> | Error-returning union (T or Error), no exceptions |

## Testing

Axon uses Catch2 v3. The test suite covers:

| Category | Test files | Tests |
|---|---|---|
| Tensor | 2 | 12 |
| Autograd | 1 | 17 |
| CPU Backend & SIMD | 5 | 32 |
| Operations & Loss | 10 | 45 |
| Modules | 8 | 22 |
| Data | 2 | 5 |
| Serialization | 1 | 6 |
| Quantization | 1 | 39 |
| Core & Benchmarks | 3 | 15 |

**193 test cases, 2401 assertions — all passing.**

Run the suite:

```bash
cmake -B build && cmake --build build && cd build && ./axon_tests
```

## License

This project is provided for educational purposes. See the repository for license details.

---

<p align="center">
  Built from scratch with ❤️ and C++20
</p>
