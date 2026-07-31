# CUDA Vector Addition

A simple CUDA program demonstrating vector addition using explicit device memory allocation.

## Overview

This project demonstrates the basic CUDA programming workflow:

- Allocate host memory
- Allocate GPU memory using `cudaMalloc()`
- Copy data from Host to Device using `cudaMemcpy()`
- Launch a CUDA kernel
- Copy results back to Host
- Free allocated memory

This example is intended for beginners learning CUDA programming.

## Directory Structure

```
vector_addition/
├── main.cu
├── README.md
└── .gitignore
```


## Concepts Covered

- CUDA Kernel
- Thread Indexing
- Grid and Block Configuration
- cudaMalloc()
- cudaMemcpy()
- cudaFree()
- Host vs Device Memory

## Future Improvements

- Unified Memory example
- CUDA Streams
- Pinned Memory
- Error checking macros
- Performance comparison
- Nsight profiling
