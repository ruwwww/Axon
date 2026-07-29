# 16 — NLL loss (Negative Log Likelihood)

## What to build

NLL loss operation: `NLL(log_probs, target) = -mean(log_probs[class_indices])`.

- Note: expects log-probabilities as input (not raw logits). This is the "NLL" half of CrossEntropy
- `NLLLossOp::forward(log_probs, target)` — looks up each sample's predicted log-prob for its class, computes mean negative
- `backward()` — gradient is `-one_hot(target) / N` (one-hot, not the softmax grad)
- Register in OpType enum and autograd dispatch
- Runtime::nll_loss() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] NLL loss forward matches hand-calculated lookup
- [ ] NLL loss backward matches finite differences
- [ ] Autograd integration test passes
- [ ] Works with 2D (N, C) and 3D (N, C, d1) inputs

## Status

ready-for-agent
