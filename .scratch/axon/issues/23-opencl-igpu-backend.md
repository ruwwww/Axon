# 23 — OpenCL 3.0 iGPU Hardware Compute Backend

## What to build

Implement an OpenCL 3.0 compute backend (`OpenCLBackend`) to offload heavy matrix multiplications (GEMM) and 2D convolution (`Conv2D`) kernels to integrated GPUs (e.g. Intel Iris Xe Graphics with 96 EUs) and discrete GPUs.

The backend should support unified shared memory (USM) / zero-copy memory mapping between host RAM and iGPU VRAM to eliminate host-device transfer overhead.

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] `OpenCLBackend` class implements `Backend` interface with OpenCL context and command queue initialization
- [ ] OpenCL kernels for FP32 GEMM and Conv2D operations compile and execute on iGPU device
- [ ] Memory allocator supports zero-copy host pointer mapping for integrated GPU memory
- [ ] Fallback mechanism gracefully falls back to CPU backend if no OpenCL 3.0 platform is detected
- [ ] All neural network forward evaluation tests pass on OpenCL backend

## Status

ready-for-agent
