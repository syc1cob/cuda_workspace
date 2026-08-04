/******************************************************************************************
 * File        : matrix_addition_ufmem.cu
 * Description :
 *      Matrix Addition using CUDA Unified Memory.
 *
 *      Concepts Covered:
 *          - cudaMallocManaged()
 *          - 2D Grid & 2D Blocks
 *          - Unified Memory
 *          - CPU Result Verification
 *          - CUDA Error Checking
 *
 * Build:
 *      nvcc matrix_addition_ufmem.cu -o matrix_addition_ufmem
 *
 * Run:
 *      ./matrix_addition_ufmem
 *
 ******************************************************************************************/

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

 // CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                   \
do                                                                         \
{                                                                          \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess)                                                \
    {                                                                      \
        std::cerr << "CUDA Error : "                                       \
                  << cudaGetErrorString(err)                               \
                  << " (" << __FILE__ << ":" << __LINE__ << ")"            \
                  << std::endl;                                            \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// GPU Kernel
__global__
void matrixAdd(const float* A,
    const float* B,
    float* C,
    int rows,
    int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}

// CPU Reference
void matrixAddCPU(const float* A,
    const float* B,
    float* C,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int index = row * cols + col;
            C[index] = A[index] + B[index];
        }
    }
}

// Initialize Matrix
void initializeMatrix(float* matrix, size_t elements)
{
    for (size_t i = 0; i < elements; i++)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Verify Result
bool verifyResult(const float* cpu,
    const float* gpu,
    size_t elements)
{
    constexpr float epsilon = 1e-5f;

    for (size_t i = 0; i < elements; i++)
    {
        if (fabs(cpu[i] - gpu[i]) > epsilon)
        {
            std::cout << "Mismatch at index "
                << i
                << "\nCPU : "
                << cpu[i]
                << "\nGPU : "
                << gpu[i]
                << std::endl;

            return false;
        }
    }

    return true;
}

// Main
int main()
{
    constexpr int ROWS = 1024;
    constexpr int COLS = 1024;

    constexpr size_t NUM_ELEMENTS = ROWS * COLS;
    constexpr size_t SIZE = NUM_ELEMENTS * sizeof(float);

    // Allocate Unified Memory
    float* A;
    float* B;
    float* C_GPU;
    float* C_CPU;

    CUDA_CHECK(cudaMallocManaged(&A, SIZE));
    CUDA_CHECK(cudaMallocManaged(&B, SIZE));
    CUDA_CHECK(cudaMallocManaged(&C_GPU, SIZE));

    // CPU reference result
    C_CPU = new float[NUM_ELEMENTS];

    // Initialize Input
    initializeMatrix(A, NUM_ELEMENTS);
    initializeMatrix(B, NUM_ELEMENTS);

    // Optional Prefetch to GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    CUDA_CHECK(cudaMemPrefetchAsync(A, SIZE, device));
    CUDA_CHECK(cudaMemPrefetchAsync(B, SIZE, device));
    CUDA_CHECK(cudaMemPrefetchAsync(C_GPU, SIZE, device));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Configure Kernel
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "\nGrid  : ("
        << blocksPerGrid.x
        << ", "
        << blocksPerGrid.y
        << ")\n";

    std::cout << "Block : ("
        << threadsPerBlock.x
        << ", "
        << threadsPerBlock.y
        << ")\n";

    // Launch Kernel
    auto gpuStart = std::chrono::high_resolution_clock::now();

    matrixAdd << <blocksPerGrid, threadsPerBlock >> > (
        A,
        B,
        C_GPU,
        ROWS,
        COLS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    // Prefetch Result Back to CPU
    CUDA_CHECK(cudaMemPrefetchAsync(
        C_GPU,
        SIZE,
        cudaCpuDeviceId));

    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU Reference
    auto cpuStart = std::chrono::high_resolution_clock::now();

    matrixAddCPU(
        A,
        B,
        C_CPU,
        ROWS,
        COLS);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    // Verify
    bool passed = verifyResult(
        C_CPU,
        C_GPU,
        NUM_ELEMENTS);

    // Timing
    auto cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart);

    auto gpuTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart);

    // Print Result
    std::cout << "\nCPU Time : "
        << cpuTime.count()
        << " ms\n";

    std::cout << "GPU Time : "
        << gpuTime.count()
        << " ms\n";

    std::cout << "\nVerification : "
        << (passed ? "PASSED" : "FAILED")
        << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C_GPU));

    delete[] C_CPU;

    CUDA_CHECK(cudaDeviceReset());

    return (passed ? EXIT_SUCCESS : EXIT_FAILURE);
}