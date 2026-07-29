<p align="center">
  <h1 align="center">Axon</h1>
  <p align="center">
    <strong>A lightweight, from-scratch neural network library in C++17</strong>
  </p>
  <p align="center">
    <em>No frameworks. No dependencies. Just pure C++.</em>
  </p>
  <p align="center">
    <a href="#features">Features</a> · <a href="#quick-start">Quick Start</a> · <a href="#architecture">Architecture</a> · <a href="#api-reference">API Reference</a> · <a href="#cifar-10-example">Example</a> · <a href="#testing">Testing</a>
  </p>
</p>

---

## Overview

Axon is a minimal but fully functional neural network library built entirely from scratch in C++17. It implements a complete deep learning pipeline — from tensors and layers to optimizers and data loading — with **zero external ML framework dependencies**.

Designed with **educational clarity** and **modularity** at its core, Axon makes the internals of neural network training transparent and accessible while remaining capable enough to train on real-world datasets like CIFAR-10.

```
Input Data ──▶ Dense ──▶ BatchNorm ──▶ ReLU ──▶ Dropout ──▶ Dense ──▶ Softmax ──▶ Predictions
                                                                          ▲
                   ◀── Backpropagation (gradient flow) ◀── Loss ◀─────────┘
```

## Features

- **Custom Tensor Engine** — Multi-dimensional array with matrix multiplication, transpose, element-wise arithmetic, and shape management
- **Layer Types** — Dense (fully connected), Activation (ReLU / Sigmoid / Softmax), Batch Normalization, Dropout
- **Optimizers** — SGD with momentum and Adam (adaptive moment estimation)
- **Loss Functions** — Cross-Entropy with numerically stable softmax-cross-entropy gradient coupling
- **Data Pipeline** — Built-in CIFAR-10 binary loader with normalization, one-hot encoding, batching, and shuffling
- **Full Backpropagation** — Automatic gradient computation through all layer types
- **Training/Inference Modes** — Proper mode switching for BatchNorm (running statistics) and Dropout (pass-through)
- **He Initialization** — Variance-scaled weight initialization for stable deep network training
- **Zero Dependencies** — Only the C++ Standard Library is required

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| C++ Compiler | C++17 support (GCC 7+, Clang 5+, MSVC 19.14+) |
| CMake | 3.14+ |

### Build

```bash
git clone <repository-url>
cd Axon
mkdir build && cd build
cmake ..
cmake --build .
```

This produces three targets:

| Target | Description |
|---|---|
| `axon` | Static library (`libaxon.a` / `axon.lib`) |
| `cifar10_train` | CIFAR-10 training example |
| `test_basic` | Unit test suite |

### Run Tests

```bash
./test_basic
```

### Train on CIFAR-10

Download the [CIFAR-10 binary dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and place the batch files in a `cifar10_real_data/` directory at the project root:

```
cifar10_real_data/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
└── test_batch.bin
```

Then run:

```bash
./cifar10_train
```

## Architecture

### Project Structure

```
Axon/
├── include/
│   └── axon/
│       ├── tensor.hpp          # Tensor class
│       ├── layers.hpp          # Layer base class & implementations
│       ├── activations.hpp     # Activation function library
│       ├── loss.hpp            # Loss functions
│       ├── optimizers.hpp      # SGD & Adam optimizers
│       ├── network.hpp         # Sequential network container
│       └── dataloader.hpp      # CIFAR-10 data loading
├── src/                        # Implementation files
├── examples/
│   └── cifar10_train.cpp       # End-to-end training example
├── tests/
│   └── test_basic.cpp          # Comprehensive unit tests
├── docs/
│   └── TESTING_REPORT.md       # Detailed test report
└── CMakeLists.txt
```

### Design Principles

- **Interface-Based Polymorphism** — All layers derive from a virtual `Layer` base class, enabling clean composition and extensibility.
- **Modular Components** — Tensors, layers, activations, losses, and optimizers are fully decoupled and independently testable.
- **Sequential Composition** — The `Network` class manages a pipeline of layers, orchestrating forward/backward passes and weight updates.
- **Cache-Based Backprop** — Each layer caches inputs/outputs during the forward pass to efficiently compute gradients during backpropagation.

### Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        Network                               │
│  ┌─────────┐  ┌───────────┐  ┌────────────┐  ┌───────────┐  │
│  │  Dense   │→│ BatchNorm  │→│ Activation  │→│  Dropout   │  │
│  └─────────┘  └───────────┘  └────────────┘  └───────────┘  │
│        ▲                                            │        │
│        └────────────── backward() ◀─────────────────┘        │
└──────────────────────────────────────────────────────────────┘
         │                                     ▲
         ▼                                     │
   ┌───────────┐                        ┌─────────────┐
   │ Optimizer  │                        │    Loss     │
   │ (SGD/Adam) │                        │(CrossEntropy)│
   └───────────┘                        └─────────────┘
```

## API Reference

### Tensor

The fundamental data structure — a multi-dimensional array backed by a flat `std::vector<float>`.

```cpp
#include "axon/tensor.hpp"

Tensor a({2, 3});                         // 2×3 zero tensor
Tensor b({2, 3}, 1.0f);                   // 2×3 tensor filled with 1.0
a.randomize_normal(0.0f, 1.0f);           // Gaussian initialization

Tensor c = a.matmul(b.transpose());       // Matrix multiplication
Tensor d = a + b;                         // Element-wise addition
Tensor e = a * 0.5f;                      // Scalar multiplication

int idx = a.argmax(0);                    // Argmax of row 0
```

### Layers

All layers implement the `Layer` interface with `forward()`, `backward()`, and optional `update_weights()`.

```cpp
#include "axon/layers.hpp"

// Fully connected: 784 inputs → 256 outputs (He initialization)
auto dense = std::make_unique<DenseLayer>(784, 256);

// Activations
auto relu    = std::make_unique<ActivationLayer>("relu");
auto sigmoid = std::make_unique<ActivationLayer>("sigmoid");
auto softmax = std::make_unique<ActivationLayer>("softmax");

// Batch normalization over 256 features
auto bn = std::make_unique<BatchNormLayer>(256);

// Dropout with 30% drop rate
auto drop = std::make_unique<DropoutLayer>(0.3f);
```

### Network

A sequential container that composes layers into a trainable pipeline.

```cpp
#include "axon/network.hpp"

Network net;
net.add(std::make_unique<DenseLayer>(3072, 512));
net.add(std::make_unique<BatchNormLayer>(512));
net.add(std::make_unique<ActivationLayer>("relu"));
net.add(std::make_unique<DropoutLayer>(0.3f));
net.add(std::make_unique<DenseLayer>(512, 10));
net.add(std::make_unique<ActivationLayer>("softmax"));

net.set_training(true);                   // Enable dropout & batch stats
Tensor output = net.forward(input);       // Forward pass
net.backward(gradient);                   // Backpropagation
```

### Loss Functions

```cpp
#include "axon/loss.hpp"

CrossEntropyLoss loss_fn;
float loss = loss_fn.compute(predictions, targets);   // Scalar loss
Tensor grad = loss_fn.gradient(predictions, targets);  // dL/dpredictions
```

### Optimizers

```cpp
#include "axon/optimizers.hpp"

// SGD with momentum
SGD sgd(/*lr=*/0.01f, /*momentum=*/0.9f);

// Adam (adaptive moment estimation)
Adam adam(/*lr=*/0.001f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*epsilon=*/1e-8f);

// Apply one optimization step
auto layers = net.get_layers();
adam.step(layers);
```

### DataLoader

Built-in support for loading CIFAR-10 binary data files.

```cpp
#include "axon/dataloader.hpp"

DataLoader loader("cifar10_real_data", /*batch_size=*/64);
loader.load_training_data();              // Load all 50,000 training images
loader.shuffle();                         // Randomize batch order

for (const auto& batch : loader.get_batches()) {
    // batch.data   → [64, 3072]  (normalized pixel values)
    // batch.labels → [64, 10]    (one-hot encoded)
}
```

## CIFAR-10 Example

A complete training script is provided in [`examples/cifar10_train.cpp`](examples/cifar10_train.cpp):

```cpp
Network net;
net.add(std::make_unique<DenseLayer>(3072, 512));
net.add(std::make_unique<BatchNormLayer>(512));
net.add(std::make_unique<ActivationLayer>("relu"));
net.add(std::make_unique<DropoutLayer>(0.3f));
net.add(std::make_unique<DenseLayer>(512, 256));
net.add(std::make_unique<BatchNormLayer>(256));
net.add(std::make_unique<ActivationLayer>("relu"));
net.add(std::make_unique<DropoutLayer>(0.2f));
net.add(std::make_unique<DenseLayer>(256, 10));
net.add(std::make_unique<ActivationLayer>("softmax"));

CrossEntropyLoss loss_fn;
Adam optimizer(0.001f);

for (int epoch = 0; epoch < 10; ++epoch) {
    net.set_training(true);
    train_loader.shuffle();

    for (const auto& batch : train_loader.get_batches()) {
        Tensor output = net.forward(batch.data);
        float loss = loss_fn.compute(output, batch.labels);
        Tensor grad = loss_fn.gradient(output, batch.labels);
        net.backward(grad);
        auto layers = net.get_layers();
        optimizer.step(layers);
    }
}
```

### Benchmark Results

With the default architecture (`3072 → 512 → 256 → 10`) trained for 10 epochs using Adam:

| Metric | Value |
|---|---|
| Training Accuracy | ~50% |
| Test Accuracy | **~47%** |
| Dataset | CIFAR-10 (50K train / 10K test) |
| Optimizer | Adam (lr = 0.001) |

> **Note:** ~47% test accuracy is a strong result for a fully-connected architecture on CIFAR-10. Convolutional networks typically achieve 90%+, but those are beyond the current scope. Random chance on CIFAR-10's 10 classes would be 10%.

## Testing

Axon includes a comprehensive test suite with **18 unit tests** covering every component:

| Category | Tests | Coverage |
|---|---|---|
| Tensor | 11 | Creation, arithmetic, matmul, transpose, reshape, sum, argmax |
| Layers | 5 | Dense, ReLU, Sigmoid, Softmax, BatchNorm, Dropout (forward & backward) |
| Loss | 1 | Cross-entropy computation and gradient |
| Optimizer | 1 | SGD and Adam parameter updates |

Run the full suite:

```bash
cd build
./test_basic
```

All 18 tests pass. See [`docs/TESTING_REPORT.md`](docs/TESTING_REPORT.md) for a detailed breakdown with performance metrics.

## Technical Details

### Numerical Stability

- **Softmax**: Log-sum-exp trick — subtracts `max(x)` before exponentiation to prevent overflow
- **Cross-Entropy**: Clamps predictions with `ε = 1e-7` to avoid `log(0)`
- **Combined Gradient**: Uses the efficient `softmax_output - target` formulation for the fused softmax + cross-entropy backward pass

### Weight Initialization

Dense layers use **He initialization**: weights are drawn from `N(0, √(2/fan_in))`, which is optimal for networks using ReLU activations.

### Batch Normalization

During training, BatchNorm normalizes using per-batch statistics and maintains exponential moving averages. During inference, it uses the accumulated running statistics for deterministic output.

### Memory Layout

Tensors use row-major storage in a contiguous `std::vector<float>`. Batch data is stored as `[batch_size, features]` matrices, enabling efficient matrix multiplication for the forward pass.

## Roadmap

Potential future enhancements:

- [ ] Convolutional layers (Conv2D, MaxPool)
- [ ] Model serialization (save/load weights)
- [ ] Data augmentation pipeline
- [ ] Performance optimizations (SIMD, multi-threading)
- [ ] Learning rate scheduling
- [ ] Additional layer types (Flatten, Residual connections)
- [ ] Extended test coverage

## License

This project is provided for educational purposes. See the repository for license details.

---

<p align="center">
  Built from scratch with ❤️ and C++17
</p>
