# Matrix Addition using CUDA Unified Memory

This project demonstrates matrix addition using **CUDA Unified Memory** (`cudaMallocManaged`). Unlike the basic matrix addition example, no explicit `cudaMemcpy()` calls are required. Memory is automatically managed by the CUDA runtime.

This example also uses **`cudaMemPrefetchAsync()`** to prefetch managed memory to the GPU before kernel execution and back to the CPU after execution, reducing page migration overhead on supported hardware.

## Concepts Covered

- CUDA Unified Memory (`cudaMallocManaged`)
- Memory Prefetching (`cudaMemPrefetchAsync`)
- 2D Grid & 2D Blocks
- CUDA Kernel Launch
- CPU Result Verification
- CUDA Error Checking

## Build

```bash
nvcc matrix_addition_ufmem.cu -o matrix_addition_ufmem
```

## Run

```bash
./matrix_addition_ufmem
```

## Expected Output

```text
CUDA Matrix Addition Example

Grid  : (64,64)
Block : (16,16)

CPU Time : 8.36743 ms
GPU Time : 0.767346 ms

Verification : PASSED
```