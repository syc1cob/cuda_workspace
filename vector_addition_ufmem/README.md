# CUDA Vector Addition using Unified Memory

This example demonstrates vector addition using **CUDA Unified Memory (Managed Memory)**.

Unlike the explicit memory version that requires manual memory copies between the CPU (Host) and GPU (Device), Unified Memory provides a single memory space that both the CPU and GPU can access.

---

## Overview

The program performs the following steps:

1. Allocate unified memory using `cudaMallocManaged()`
2. Initialize input vectors on the CPU
3. Launch a CUDA kernel to perform vector addition
4. Synchronize using `cudaDeviceSynchronize()`
5. Perform the same computation on the CPU
6. Compare CPU and GPU results
7. Release unified memory

---

## Directory Structure

```
vector_addition_ufmem/
├── kernel.cu
├── README.md
└── vector_addition_ufmem.vcxproj
```

---

## CUDA Concepts Covered

- CUDA Kernel Launch
- Thread Indexing
- Grid and Block Configuration
- Unified Memory
- `cudaMallocManaged()`
- `cudaDeviceSynchronize()`
- CPU vs GPU Result Verification

---

## Kernel

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

Each CUDA thread computes one element of the output vector.

---

## Unified Memory Allocation

Instead of using:

```cpp
cudaMalloc()
cudaMemcpy()
```

this example uses:

```cpp
cudaMallocManaged(&A, vectorLength * sizeof(float));
cudaMallocManaged(&B, vectorLength * sizeof(float));
cudaMallocManaged(&C, vectorLength * sizeof(float));
```

Unified Memory automatically migrates memory pages between the CPU and GPU when required.

---

## Kernel Launch

```cpp
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);

vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
```

The kernel launches enough blocks so that every vector element is processed.

---

## Synchronization

```cpp
cudaDeviceSynchronize();
```

The CPU waits until the GPU has completed execution before accessing the results.

---

## Result Verification

The GPU result is compared against a CPU implementation.

```cpp
serialVecAdd(A, B, comparisonResult, vectorLength);
```

If all elements match within a small floating-point tolerance, the program prints:

```
Unified Memory: CPU and GPU answers match
```

---

## Build

### Visual Studio 2022

1. Open `vector_addition_ufmem.sln`
2. Select **x64**
3. Build the solution
4. Run the executable

---

## Example Output

```
Unified Memory: CPU and GPU answers match
```

---

## Explicit Memory vs Unified Memory

| Explicit Memory | Unified Memory |
|-----------------|----------------|
| `cudaMalloc()` | `cudaMallocManaged()` |
| Requires `cudaMemcpy()` | No explicit copies |
| Separate Host and Device memory | Single managed memory space |
| More control | Easier to program |
| Preferred for maximum performance | Excellent for learning and rapid development |

---

## Learning Objectives

After completing this example, you should understand:

- What Unified Memory is
- How `cudaMallocManaged()` works
- Why `cudaDeviceSynchronize()` is required
- Differences between explicit and managed memory
- CPU/GPU shared address space

---

## Next Steps

The next CUDA topics to explore are:

- CUDA Streams
- Asynchronous Memory Copy
- Pinned Host Memory
- Shared Memory
- CUDA Events
- Matrix Multiplication
- Reduction
- Nsight Systems
- Nsight Compute

---

## References

- NVIDIA CUDA C++ Programming Guide
- CUDA Runtime API Documentation