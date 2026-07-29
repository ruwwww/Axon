# 21 — Replace repeated OpType switch with table-based dispatch

## What to build

The `Autograd::backward()` method in `src/autograd/autograd.cpp:1118-1159` contains a 10-case `switch(node.op)` that dispatches each operation type to its `backward()` function. Every new op requires adding a case to this switch.

Replace the switch with a dispatch table (e.g., `std::array` or `std::unordered_map` of function pointers indexed by `OpType`). Then registering a new op is just adding an entry to the table.

This is a mechanical refactor. No behavior change.

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Switch statement replaced with table-based dispatch
- [ ] Every existing op has an entry in the dispatch table
- [ ] All existing tests pass unchanged
- [ ] Adding a new op only requires adding one entry (not editing the switch + enum + forward)

## Status

ready-for-agent
