# 19 — Replace static TensorId counter with per-Runtime counter

## What to build

The current `Tensor::next_id()` uses a `static TensorId counter` — global state that violates the "no global state" rule and means Tensor IDs are shared across all Runtime instances.

Move ID generation to Runtime so each Runtime has its own counter:

- Add `std::atomic<TensorId> next_tensor_id_` to Runtime (initialized to 0)
- Add `TensorId Runtime::next_tensor_id()` method
- Change Tensor constructor to accept a Runtime& and use its counter
- Tensor factory methods already take Runtime& — pass it through

This is a mechanical refactor. No behavior change for single-Runtime usage.

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Two separate Runtime instances produce non-overlapping ID sequences (Runtime A: 0,1,2... Runtime B: 0,1,2...)
- [ ] No `static` variables remain for ID generation
- [ ] All existing tests pass unchanged

## Status

ready-for-agent
