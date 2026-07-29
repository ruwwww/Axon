# Error-return types instead of exceptions across subsystem boundaries

All cross-subsystem public APIs return `Expected<T>` or `Status` rather than throwing exceptions. Exceptions are reserved for unrecoverable internal bugs within a subsystem. This keeps failure paths explicit and avoids the hidden control flow that makes exception-based error handling hard to reason about in a framework with many chained operations.
