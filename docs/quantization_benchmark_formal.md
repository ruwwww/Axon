# Comprehensive Benchmark: Post-Training Quantization (FP32 vs Q4_0 4-Bit) in Axon

*Author: Antigravity AI Pair Programmer*  
*Date: July 29, 2026*

---

## 🎯 Benchmark Purpose

To formally measure the impact of **Axon's `Q4_0` 4-Bit Block Quantization** on model accuracy retention, memory footprint reduction, and inference throughput when deployed on real unseen image data.

---

## 🔬 Experimental Setup

1. **Dataset**: Official **10,000 Unseen MNIST Test Images** (`datasets/mnist`).
2. **Model**: 2-Layer Multilayer Perceptron (`Linear(784 -> 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128 -> 10)`).
3. **Training & Quantization Method**:
   - Model was trained on real MNIST images in `Float32`.
   - Weights were quantized post-training into **`Q4_0` 4-bit blocks** (32 elements per block with scale factors $\delta$) using `cpu::quantize()`.
   - Evaluated using both full precision `Float32` baseline and `Q4_0` 4-bit block quantization.

---

## 📊 Comprehensive Benchmark Results

| Performance Metric | Full Precision (`Float32`) | 4-Bit Quantized (`Q4_0`) | Measured Delta / Compression |
|---|---|---|---|
| **Test Accuracy (10,000 Unseen Images)** | **83.31%** | **81.28%** | **Only 2.03% Accuracy Drop!** |
| **Weight Buffer RAM Footprint** | 401,408 Bytes (401.4 KB) | **56,448 Bytes** (56.4 KB) | **$7.11\times$ RAM Memory Savings** |
| **Total Test Inference Latency** | `3,071.71 ms` | **`3,030.43 ms`** | **1.01x faster** |
| **CPU Processing Throughput** | 3,255.52 img/sec | **3,299.87 img/sec** | Equivalent CPU throughput |

---

## 💡 Key Architectural Insights

1. **High Accuracy Retention (97.5% Accuracy Preserved)**:
   - When evaluated on **real trained model weights**, 4-bit block quantization (`Q4_0`) achieves **81.28% test accuracy**, suffering only a minor **2.03% accuracy drop** compared to full `Float32` precision (83.31%).
   - *Why did the earlier synthetic test show a higher error?* Random Gaussian noise tensors lack the structured clustering of real trained neural weights, making random matrices far more sensitive to quantization bounds than real model layers.

2. **$7.11\times$ Memory Footprint Reduction**:
   - Shrinks the layer weight buffers down from **401.4 KB to 56.4 KB**, reducing RAM requirements by over **85.9%**.

3. **Inference Performance**:
   - On-the-fly block dequantization enables compressed model inference with equivalent CPU throughput (~3,300 images/sec).
