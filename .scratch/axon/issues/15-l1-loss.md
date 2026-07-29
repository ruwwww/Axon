# 15 — L1 loss (Mean Absolute Error)

## What to build

L1 loss operation: `L1(input, target) = mean(|input - target|)`.

- `L1LossOp::forward()` — computes mean absolute error
- `backward()` — gradient is `sign(input - target) / N` (sign function, +1 for positive, -1 for negative)
- Register in OpType enum and autograd dispatch
- Runtime::l1_loss() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] L1 loss forward matches hand-calculated value
- [ ] L1 loss backward matches finite differences
- [ ] Autograd integration test passes

## Status

ready-for-agent
