/******************************************************************************************
 * File        : matrix_addition.cu
 * Author      : Syalu C S
 * Description :
 *      Matrix Addition using CUDA
 *
 *      This example demonstrates:
 *          - 2D Grid and 2D Block configuration
 *          - GPU kernel execution
 *          - CUDA memory allocation
 *          - Host <-> Device memory copy
 *          - CPU reference implementation
 *          - Result verification
 *          - CUDA error handling
 *
 * Build:
 *      nvcc matrix_addition.cu -o matrix_addition
 *
 * Run:
 *      ./matrix_addition
 *
 ******************************************************************************************/

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

 // CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                     \
do                                                                           \
{                                                                            \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess)                                                  \
    {                                                                        \
        std::cerr << "\nCUDA ERROR\n";                                       \
        std::cerr << "File      : " << __FILE__ << std::endl;                \
        std::cerr << "Line      : " << __LINE__ << std::endl;                \
        std::cerr << "Error Code: " << err << std::endl;                     \
        std::cerr << "Reason    : " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

// GPU Kernel
__global__
void matrixAdd(const float* A,
    const float* B,
    float* C,
    int rows,
    int cols)
{
    // Compute row index handled by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute column index handled by this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}

// CPU Reference Implementation
void matrixAddCPU(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
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
void initializeMatrix(std::vector<float>& matrix)
{
    for (auto& value : matrix)
    {
        value = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Verify GPU Result
bool verifyResult(const std::vector<float>& cpu,
    const std::vector<float>& gpu)
{
    constexpr float epsilon = 1e-5f;

    for (size_t i = 0; i < cpu.size(); i++)
    {
        if (std::fabs(cpu[i] - gpu[i]) > epsilon)
        {
            std::cout << "\nMismatch at index "
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

    // Allocate Host Memory
    std::vector<float> h_A(NUM_ELEMENTS);
    std::vector<float> h_B(NUM_ELEMENTS);

    std::vector<float> h_C_CPU(NUM_ELEMENTS);
    std::vector<float> h_C_GPU(NUM_ELEMENTS);

    initializeMatrix(h_A);
    initializeMatrix(h_B);

    // Allocate Device Memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, SIZE));
    CUDA_CHECK(cudaMalloc(&d_B, SIZE));
    CUDA_CHECK(cudaMalloc(&d_C, SIZE));

    // Copy Input Data to Device
    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A.data(),
        SIZE,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B.data(),
        SIZE,
        cudaMemcpyHostToDevice));

    // Configure Kernel Launch
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "\nLaunching Kernel...\n";

    std::cout << "Grid  : ("
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
        d_A,
        d_B,
        d_C,
        ROWS,
        COLS);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    // Copy Result Back
    CUDA_CHECK(cudaMemcpy(
        h_C_GPU.data(),
        d_C,
        SIZE,
        cudaMemcpyDeviceToHost));

    // CPU Reference
    auto cpuStart = std::chrono::high_resolution_clock::now();

    matrixAddCPU(
        h_A,
        h_B,
        h_C_CPU,
        ROWS,
        COLS);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    // Verify
    bool passed = verifyResult(
        h_C_CPU,
        h_C_GPU);

    // Timing
    auto cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart);

    auto gpuTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart);

    // Print Results
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
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaDeviceReset());

    return (passed ? EXIT_SUCCESS : EXIT_FAILURE);
}