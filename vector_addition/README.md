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
├── CMakeLists.txt
├── main.cu
├── README.md
└── .gitignore
```

## Build

```bash
mkdir build
cd build

cmake ..
make
```

## Run

Default vector size:

```bash
./vector_addition
```

Specify vector length:

```bash
./vector_addition 4096
```

Example:

```bash
./vector_addition 1000000
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