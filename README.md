<p align="center">
  <h1 align="center">Axon</h1>
  <p align="center">
    <strong>A minimal deep learning framework in C++20</strong>
  </p>
  <p align="center">
    <em>Eager autograd, CPU backend, GGML-style quantization — built from scratch.</em>
  </p>
  <p align="center">
    <a href="#features">Features</a> · <a href="#quick-start">Quick Start</a> · <a href="#architecture">Architecture</a> · <a href="#api-reference">API Reference</a> · <a href="#examples">Examples</a>
  </p>
</p>

---

## Overview

Axon is a minimal deep learning framework in C++20 that implements a complete training pipeline — tensors, autograd, neural network modules, optimizers, data loading, serialization, and quantization — with **zero external ML framework dependencies**.

Designed to show how modern AI runtimes are built: eager execution, per-operation autograd recording, a clean separation between backend kernels and graph mechanics, and quantized inference formats inspired by GGML.

The framework is capable enough to train a ResNet-18 on CIFAR-10 and run quantized INT4 inference.

## Features

- **Eager Autograd** — Operations record a linear graph during forward; backward replays in reverse with gradient accumulation
- **Tensor Engine** — Multi-dimensional arrays with shared ownership (`shared_ptr<Storage>`), strided access, and dtype/device/quantization metadata
- **Neural Network Modules** — Linear, Conv2D, BatchNorm, LayerNorm, Dropout, Flatten, Embedding, Residual, Sequential
- **ResNet-18** — Complete implementation with BasicBlock and configurable output classes
- **Optimizers** — SGD (with momentum) and AdamW
- **Loss Functions** — Cross-entropy (fused log-softmax + NLL) and MSE
- **Data Pipeline** — CIFAR-10 and MNIST loaders with batching and shuffling
- **GGML-style Quantization** — Q8_0 (8-bit) and Q4_0 (4-bit) block-wise quantization for inference
- **Serialization** — Save/load individual tensors and full model checkpoints
- **Error Handling** — `Expected<T>` return types across all subsystem boundaries (no exceptions)
- **CPU Backend** — Reference kernels for matmul, convolutions, pooling, normalization, and element-wise ops
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

All **105 tests pass** (688 assertions).

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
│   ├── autograd/           # Graph, GraphNode, OpType, GradientMap, operation structs
│   ├── backend/            # cpu_backend.h (kernel declarations)
│   ├── core/               # Expected<T>, types (DType, Device, QuantFormat), serialization
│   ├── data/               # Dataset interface, DataLoader, CIFAR10, MNIST
│   ├── nn/                 # Module base, Parameter, Sequential, Linear, Conv2D, BatchNorm,
│   │                       # LayerNorm, Dropout, Flatten, Embedding, Residual, ResNet18,
│   │                       # SGD, AdamW, CrossEntropyLossOp, MSELossOp
│   ├── runtime/            # Runtime (public API), Allocator
│   ├── storage/            # Storage (aligned memory), QuantizationDescriptor
│   └── tensor/             # Tensor, TensorType
├── src/                    # Implementation files
├── tests/                  # Catch2 v3 test suite (24 files, 105 tests)
├── examples/
│   ├── cifar10.cpp         # ConvNet training on CIFAR-10
│   ├── mnist.cpp           # Linear classifier on MNIST
│   └── mnist_benchmark_minimal.cpp
├── docs/
│   ├── adr/                # Architecture Decision Records
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
   │  Allocator │  │  Backend     │  │  Autograd    │
   │ (creates   │  │ (cpu::       │  │ (owns Graph  │
   │  Storage)  │  │  matmul,     │  │  + Gradient- │
   └────────────┘  │  relu, ...)  │  │  Map)        │
                   └──────────────┘  └──────┬───────┘
                                            │
                                     ┌──────┴──────┐
                                     │    Graph     │
                                     │  GraphNode[] │
                                     │  each node:  │
                                     │  op, inputs, │
                                     │  output,     │
                                     │  runtime*,   │
                                     │  op_data     │
                                     └─────────────┘
```

### Key design decisions

- **Runtime is the public API** — not a scheduler or executor. Operations call into the backend and record graph nodes themselves.
- **Operations own graph recording** — each operation's `forward()` checks `requires_grad`, allocates output, calls the backend kernel, and appends a `GraphNode` if needed.
- **Backend is a namespace of free functions** — `cpu::matmul(out, a, b)`. Receives raw `Tensor*` pointers. Never accesses the Graph. Adding a GPU backend means adding `cuda::matmul(...)`.
- **Tensor is a lightweight frontend** — holds `shared_ptr<Storage>` and an immutable `TensorType` descriptor. Copyable, reference-counted memory.
- **No exceptions** — all fallible operations return `Expected<T>`, a `std::variant<T, Error>`.
- **No global state** — Allocator is owned by Runtime. No singletons.

### Domain terminology

| Term | Meaning |
|---|---|
| Tensor | Lightweight copyable handle to n-dimensional data |
| Storage | Reference-counted aligned memory block |
| TensorType | Immutable shape/strides/dtype/device/quant descriptor |
| Runtime | Execution context (allocator, autograd, training mode) |
| Operation | Stateless functor with `forward()` and `backward()` |
| Graph | Linear sequence of GraphNodes recorded during forward |
| GraphNode | One recorded operation: op type, inputs, output, runtime*, op_data |
| Autograd | Owns Graph + GradientMap; drives backward traversal |
| Module | Base class for neural network building blocks |
| Parameter | Wraps a Tensor with gradient and trainable flag |
| Expected\<T\> | Error-returning union (T or Error), no exceptions |

## API Reference

### Tensor

```cpp
#include "axon/tensor/tensor.h"

Runtime rt;
auto a = Tensor::randn(rt, {2, 3});       // 2×3 random normal
auto b = Tensor::zeros(rt, {2, 3});       // 2×3 zeros
auto c = Tensor::ones(rt, {3, 4});        // 3×4 ones

float* ptr = a.data<float>();              // raw pointer to data
auto shape = a.type().shape();             // {2, 3}
a.set_requires_grad(true);                 // enable gradient tracking
```

### Runtime (public API)

```cpp
#include "axon/runtime/runtime.h"

Runtime rt;

// Tensor creation
auto x = rt.randn({32, 784});              // random normal
auto w = rt.randn({784, 256});             // random normal
auto y = rt.zeros({32, 256});              // zeros

// Operations (autograd-enabled)
auto z = rt.matmul(x, w);                   // Expected<Tensor>
auto a = rt.relu(z.value());                // Expected<Tensor>
auto s = rt.add(a.value(), y);              // Expected<Tensor>

// Convolution / pooling
auto c = rt.conv2d(input, weight, bias, 1, 1);
auto p = rt.maxpool2d(c.value(), 3, 2);
auto q = rt.avgpool2d(c.value(), 3, 2);
```

### Module & Sequential

```cpp
#include "axon/nn/sequential.h"
#include "axon/nn/linear.h"
#include "axon/nn/conv2d.h"
#include "axon/nn/batchnorm.h"
#include "axon/nn/dropout.h"
#include "axon/nn/flatten.h"

Runtime rt;

Sequential model;
model.add(std::make_unique<Conv2D>(rt, 3, 16, 3, 1, 1));
model.add(std::make_unique<BatchNorm>(rt, 16));
model.add(std::make_unique<Dropout>(0.2f));
model.add(std::make_unique<Flatten>());
model.add(std::make_unique<Linear>(rt, 16 * 32 * 32, 10));

model.train();                              // enable training mode
auto output = model.forward(rt, input);     // Expected<Tensor>
model.eval();                               // inference mode
```

### Parameter

```cpp
#include "axon/nn/parameter.h"

Parameter p(Tensor::randn(rt, {256, 10}), true);
p.tensor()                                // the wrapped Tensor
p.grad()                                  // gradient Tensor (lazily allocated)
p.trainable()                             // bool
```

### Loss Functions

```cpp
#include "axon/nn/cross_entropy.h"

auto loss = CrossEntropyLossOp::forward(rt, logits, targets);   // Expected<Tensor>
float loss_value = loss.value().data<float>()[0];
```

### Optimizers

```cpp
#include "axon/nn/sgd.h"
#include "axon/nn/adamw.h"

SGD sgd(rt, model.parameters(), 0.01f, 0.9f);           // SGD with momentum
AdamW adam(rt, model.parameters(), 0.001f);               // AdamW

optimizer.zero_grad();      // zero all parameter gradients
// ... forward + backward ...
optimizer.step();            // update parameters
```

### ResNet-18

```cpp
#include "axon/nn/resnet18.h"

Runtime rt;
ResNet18 resnet(rt, 10);                    // 10 classes (CIFAR-10)

auto output = resnet.forward(rt, input);    // Expected<Tensor>
auto params = resnet.parameters();          // all trainable parameters
```

### Data Loading

```cpp
#include "axon/data/cifar10.h"
#include "axon/data/dataloader.h"

CIFAR10 dataset(rt, "cifar10_real_data", true);   // training split
DataLoader loader(dataset, 64, true);              // batch size 64, shuffle

for (auto& batch : loader.iter()) {
    // batch.inputs   → Tensor [64, 3, 32, 32]
    // batch.targets  → Tensor [64, 10]  (one-hot)
}
```

### Serialization

```cpp
#include "axon/core/serialize.h"

save_tensor(tensor, "weight.bin");                     // save one tensor
auto t = load_tensor(rt, "weight.bin");                // load one tensor

save_checkpoint(model, "checkpoint.axon");             // save all parameters
load_checkpoint(rt, model, "checkpoint.axon");         // load all parameters
```

### Quantization

```cpp
#include "axon/storage/quantization.h"

QuantizationDescriptor q8{QuantFormat::Q8_0, 32};     // 8-bit, block size 32
QuantizationDescriptor q4{QuantFormat::Q4_0, 32};     // 4-bit, block size 32

// Quantized tensors for inference (no autograd)
auto q_type = TensorType::contiguous({1024, 768}, DType::Float32, Device::CPU);
q_type = TensorType(q_type.shape(), q_type.strides(), DType::Float32, Device::CPU, QuantFormat::Q8_0);

void* quantized_data = quantize_q8_0(raw_data, num_blocks);
void* output = dequantize_q8_0(quantized_data, num_blocks);
```

### Error Handling

```cpp
#include "axon/core/expected.h"

auto result = rt.matmul(a, b);
if (!result) {
    std::cerr << "matmul failed: " << result.error().message << std::endl;
    return 1;
}
Tensor& output = result.value();    // safe — checked above
```

All operations that can fail return `Expected<T>`. The `RETURN_IF_ERROR(expr)` macro propagates errors:

```cpp
RETURN_IF_ERROR(cpu::matmul(out, a, b));
```

## Examples

### MNIST (Linear Classifier)

[`examples/mnist.cpp`](examples/mnist.cpp) — A single Linear layer (784→10) trained with SGD:

```
Epoch 1/3  Avg Loss: 0.5231
Epoch 2/3  Avg Loss: 0.3124
Epoch 3/3  Avg Loss: 0.2987
```

### CIFAR-10 (Small ConvNet)

[`examples/cifar10.cpp`](examples/cifar10.cpp) — A two-layer ConvNet (Conv2D→ReLU→Conv2D→ReLU→Flatten→Linear) trained with AdamW.

### ResNet-18 on CIFAR-10

The `ResNet18` module in [`include/axon/nn/resnet18.h`](include/axon/nn/resnet18.h) is a complete ResNet-18 implementation with BasicBlock, shortcut connections, and configurable output classes. Train it using the same pattern as the CIFAR-10 example.

## Quantization

Axon supports GGML-style block-wise quantization for inference:

| Format | Bits/weight | Block size | Block layout |
|---|---|---|---|
| Q8_0 | 8 | 32 | 1 fp16 scale + 32 × int8 |
| Q4_0 | 4 | 32 | 1 fp16 scale + 16 × int4 (packed) |

Quantization operates on blocks of 32 weights. Each block has one fp16 scale factor and the quantized weights. This is the same scheme used by GGML for CPU inference.

Benchmark results are in [`docs/quantization_benchmark_formal.md`](docs/quantization_benchmark_formal.md).

## Testing

Axon uses Catch2 v3. The test suite covers:

| Category | Test files | Tests |
|---|---|---|
| Tensor | 1 | 6 |
| Autograd | 1 | 17 |
| CPU Backend | 2 | 19 |
| Operations | 8 | 23 |
| Modules | 8 | 22 |
| Data | 2 | 5 |
| Serialization | 1 | 6 |
| Quantization | 1 | 5 |
| Expected | 1 | 5 |

**105 test cases, 688 assertions — all passing.**

Run the suite:

```bash
cmake -B build && cmake --build build && cd build && ./axon_tests
```

## Roadmap

- GELU, Reshape, Transpose, Slice, Mean, Sum, Max operations
- L1 loss, NLL loss
- Vanilla Adam optimizer
- Additional quantization formats (Q6_K, Q5_K, Q4_K, Q3_K, Q2_K)
- Table-based autograd dispatch
- Per-Runtime TensorId counter

## License

This project is provided for educational purposes. See the repository for license details.

---

<p align="center">
  Built from scratch with ❤️ and C++20
</p>
