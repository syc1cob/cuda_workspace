# Matrix Addition using CUDA

This project demonstrates matrix addition using **CUDA C++**. Each GPU thread computes one element of the output matrix using a **2D Grid** and **2D Thread Blocks**.

## Features

- CUDA kernel programming
- 2D Grid & 2D Block configuration
- Host and Device memory allocation
- CPU result verification
- CUDA error checking

## Build

```bash
nvcc matrix_addition.cu -o matrix_addition
```

## Run

```bash
./matrix_addition
```

## Expected Output

```text
CUDA Matrix Addition Example

Grid  : (64,64)
Block : (16,16)

CPU Time :  12.5854 ms
GPU Time : 1.575 ms

Verification : PASSED
```

## Concepts Covered

- CUDA Kernel
- Grid, Block, Thread
- 2D Thread Indexing
- `cudaMalloc()`
- `cudaMemcpy()`
- `cudaDeviceSynchronize()`
- `cudaFree()`

## Next Examples

- Matrix Multiplication
- Shared Memory
- Constant Memory
- Parallel Reduction
- Thrust
- CUB