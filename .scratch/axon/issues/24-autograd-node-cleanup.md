# 24 — Autograd Node Refactoring and Input Cleanup

## What to build

Refactor autograd Node base class and concrete subclasses:
1. Modify include/axon/autograd/node.h so virtual std::vector<Tensor>& inputs() = 0; and virtual const std::vector<Tensor>& inputs() const = 0; return non-const and const references appropriately.
2. Remove const_cast from src/autograd/autograd.cpp in Autograd::backward().
3. Refactor all 15 concrete Node subclasses in include/axon/autograd/nodes.h and src/autograd/autograd.cpp so they store input tensors only inside inputs_ (e.g., inputs_[0], inputs_[1]), removing duplicate member fields (a_, b_, etc.).
4. Remove #include "axon/autograd/nodes.h" from include/axon/autograd/autograd.h so autograd.h only includes node.h.
5. Run cmake --build build --config Release --target axon_tests and .\build\Release\axon_tests.exe to verify all 190 test cases pass cleanly.

## Acceptance criteria

- [x] node.h has non-const and const inputs() member functions
- [x] Autograd::backward() uses inputs() non-const overload without const_cast
- [x] All 15 concrete Node subclasses store inputs only in inputs_ vector
- [x] autograd.h includes node.h and does not include nodes.h
- [x] All 190 unit test cases pass cleanly

## Status

completed
