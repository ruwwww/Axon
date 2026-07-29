# 06 — Serialization

## Context

Axon is a minimal deep learning framework. This ticket builds the serialization layer for saving and loading tensors and model checkpoints. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture. Tickets 01 and 02 must be complete (need Tensor and ops for checkpoints).

## What to build

A thin serialization layer of free functions:

- **`.axon` binary format**:

```
[Magic "AXON"][Version: uint32]
[TensorCount: uint32]
for each tensor:
    [NameLen: uint32][Name: char * NameLen]
    [DType: uint32][QuantType: uint32]
    [NDims: uint32][Shape: uint64 * NDims]
    [DataBytes: uint64][Data: uint8 * DataBytes]
```

- DataBytes is the exact Storage::size_bytes, including packed quantized layouts.
- No GGUF parsing, no GGUF header compatibility. The `.axon` format is framework-private.

- **`save_tensor(tensor, path)`**: Writes a single tensor to `.axon` format.
- **`load_tensor(runtime, path)`**: Reads header, allocates Storage via `runtime.allocator()`, copies raw bytes into storage->data, returns Tensor.
- **`save_checkpoint(module, path)`**: Iterates `module.parameters()`, writes each (parameter name, tensor) pair.
- **`load_checkpoint(runtime, module, path)`**: Reads saved tensors, matches by name to registered parameters, overwrites each parameter's Storage.
- **Serialization roundtrip tests**: Save a Module → load into fresh instance → compare parameters bit-for-bit.

## Key interfaces

```cpp
// axon/serialize.h
Expected<void> save_tensor(const Tensor& t, const std::string& path);
Expected<Tensor> load_tensor(Runtime& rt, const std::string& path);

Expected<void> save_checkpoint(const Module& m, const std::string& path);
Expected<void> load_checkpoint(Runtime& rt, Module& m, const std::string& path);
```

## Acceptance criteria

- [ ] `save_tensor` / `load_tensor` roundtrips correctly: tensor data is identical after save/load.
- [ ] `.axon` file format matches the spec (magic, version, counts).
- [ ] `save_checkpoint` / `load_checkpoint` roundtrips all parameters of a Module.
- [ ] Loading into a different Module instance restores all parameter values.
- [ ] Roundtrip works for tensors with different dtypes (float32, int32) and shapes.
- [ ] Error on missing file returns Expected error, not a crash.

## Blocked by

02 — Operations + Graph + Autograd (needs ops to build a Module with parameters)

## Status

ready-for-agent
