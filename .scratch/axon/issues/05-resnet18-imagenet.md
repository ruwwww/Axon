# 05 — ResNet18 + ImageNet

## Context

Axon is a minimal deep learning framework. This ticket builds the full ResNet18 architecture and ImageNet training pipeline. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture. Tickets 01–04 must be complete.

## What to build

The main goal of Phase 1 — a complete ResNet18 training pipeline on ImageNet:

- **ResNet18 Module**: Composed of:
  - Initial Conv2D (3→64, 7x7, stride 2) + BatchNorm + ReLU + MaxPool (3x3, stride 2)
  - Layer 1: [BasicBlock(64→64) x 2] — no stride
  - Layer 2: [BasicBlock(64→128, stride 2), BasicBlock(128→128)] — conv3x3 + conv3x3 with skip connection
  - Layer 3: [BasicBlock(128→256, stride 2), BasicBlock(256→256)]
  - Layer 4: [BasicBlock(256→512, stride 2), BasicBlock(512→512)]
  - Average Pool + Flatten + Linear(512→1000)
  - BasicBlock: 3x3 conv → BN → ReLU → 3x3 conv → BN → skip (1x1 conv if dimensions change) → ReLU
- **ImageNet Dataset**: Pre-downloaded directory structure with subdirectories per class. Reads and decodes raw images. Phase 1: use pre-processed tensors or simple binary format if raw image parsing is complex.
- **Training pipeline**: Multi-epoch training loop with AdamW optimizer, CrossEntropy loss, learning rate scheduling (step decay).
- **Accuracy metrics**: Top-1 and Top-5 accuracy computation.
- **Integration test**: Train for a small number of steps and verify loss decreases. Full ImageNet convergence is the acceptance gate for Phase 1 completion.

## Acceptance criteria

- [ ] ResNet18 module compiles and produces correct output shape (N x 1000) for a random input.
- [ ] All BasicBlocks forward correctly with skip connections matching at dimension boundaries.
- [ ] ImageNet dataset iterates correctly with DataLoader.
- [ ] Training loop for N steps produces decreasing loss.
- [ ] Memory: training with batch size ≥ 32 fits within reasonable RAM.

## Blocked by

04 — Conv2D + AdamW + Normalization + CIFAR10

## Status

ready-for-agent
