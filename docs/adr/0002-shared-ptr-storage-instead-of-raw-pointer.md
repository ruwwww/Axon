# Shared ownership of Storage via shared_ptr instead of non-owning raw pointer

The original spec defined Tensor as holding a non-owning `Storage*`. Phase 1 instead uses `std::shared_ptr<Storage>` within Tensor. The small refcount overhead on copy is acceptable because it eliminates dangling-pointer risk and the complexity of an external lifetime manager. A future phase can restore the non-owning model for zero-cost views if profiling shows the refcount is a bottleneck.
