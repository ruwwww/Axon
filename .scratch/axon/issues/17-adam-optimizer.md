# 17 — Vanilla Adam optimizer

## What to build

Vanilla Adam optimizer (without weight decay, separate from the existing AdamW).

- `Adam` class following same pattern as `AdamW`:
  - Constructor: `Adam(Runtime& rt, std::vector<Parameter*> params, float lr, float beta1=0.9, float beta2=0.999, float eps=1e-8)`
  - State: `m` (1st moment), `v` (2nd moment) per parameter
  - `step()`: bias-corrected Adam update without weight decay
  - `zero_grad()`: zeros parameter gradients
  - Eager state allocation on construction
- Tests: single Parameter with known gradient, verify update matches hand calculation

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Adam step applies correct bias-corrected update formula
- [ ] State tensors are allocated eagerly
- [ ] zero_grad zeros all gradients
- [ ] Unit test: update matches hand-calculated value for known inputs

## Status

ready-for-agent
