#pragma once

// CUDA 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUBLAS
#include <cublas_v2.h>
#include <cublasLt.h>

// STL
#include <type_traits>
#include <vector>
#include <immintrin.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <cstring>

#include <benchmark.h>

#define BENCHMARK 1

#if defined(_WIN32) || defined(_WIN64)
#define OS_WINDOWS 1
#define OS_LINUX 0
#elif defined(__linux__)
#define OS_WINDOWS 0
#define OS_LINUX 1
#else
#define OS_WINDOWS 0
#define OS_LINUX 0
#endif

using std::cout;
using std::clog;
using std::endl;
using std::cerr;
using std::flush;

namespace cudavec {
    /**
     * \brief Struct for storing memory statistics of current device
     **/
    struct DeviceMemoryStatus {
        DeviceMemoryStatus();
        size_t mFreeAmount;
        size_t mTotalAmount;
        size_t mUsedAmount;
    };

    __host__ std::ostream& operator<<(std::ostream& stream, const DeviceMemoryStatus& memStatus);

    /**
     * \brief Empty kernel for lazy loading
     **/
    __global__ void KernelWarmup();

    template<typename Ty_>
    __global__ void addKernel(Ty_* c, const Ty_* a, const Ty_* b, uint32_t size);

    template<typename Ty_>
    __global__ void mulKernel(Ty_* c, const Ty_* a, const Ty_* b, uint32_t size);

    template<typename Ty_>
    __global__ void divKernel(Ty_* c, const Ty_* a, const Ty_* b, uint32_t size);

    template<typename Ty_>
    __global__ void addEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, uint32_t size);

    template<typename Ty_>
    __global__ void mulEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, uint32_t size);

    template<typename Ty_>
    __global__ void divEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, uint32_t size);

    /**
     * \brief Matrix multiplication kernel (no optimization for equal dim matrices)
     **/
    template <typename Ty_>
    __global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, uint32_t M, uint32_t N, uint32_t K);

    /**
     * \brief Host function for initializing CUDA contexes
     **/
    __host__ void CUDAContextInit(int device);

    __host__ std::ostream& operator<<(std::ostream& stream, const cudaDeviceProp& devProps);

    __host__ uint32_t deviceSharedMemory(const int& device);

    /**
     * \brief Performs matrix multiplication using CPU on two matrices
     * \param a pointer to a 1D serialized array with size of M*K
     * \param b pointer to a 1D serialized array with size of N*K
     * \param M array's (uncommon) dimension
     * \param N array's (uncommon) dimension
     * \param K both arrays' common dimensions
     **/
    template<typename Ty_>
    __host__ std::vector<Ty_> matmul_flat(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K);

#if OS_WINDOWS
    #include <windows.h>
    /**
     * \brief Performs matrix multiplication using AVX instruction set on two matrices
     * \param a pointer to a 1D serialized array with size of M*K
     * \param b pointer to a 1D serialized array with size of N*K
     * \param M array's (uncommon) dimension
     * \param N array's (uncommon) dimension
     * \param K both arrays' common dimensions
     **/
    template<typename Ty_>
    __host__ std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K);

    uint64_t getTotalSystemMemory();
#elif 
    // TODO implement avx for linux specific idk
#include <unistd.h>
    unsigned long long getTotalSystemMemory()
#endif
    /**
     * \brief Perform an operator on two vectors
     * \param a first operand vector
     * \param b second operand vector
     * \param kernelFunction kernel operator function to be called
     **/
    template <typename Ty_, typename KernelFunc>
    __host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction);

    /**
     * \brief Perform an operator on a vector with an constant operand value
     * \param a first operand vector
     * \param b operand value
     * \param kernelFunction kernel operator function to be called
     **/
    template <typename Ty_, typename KernelFunc>
    __host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction);

    /**
     * \brief Performs matrix multiplication using CUDA on two matrices
     * \param a pointer to a 1D serialized array with size of M*K
     * \param b pointer to a 1D serialized array with size of N*K
     * \param M array's (uncommon) dimension
     * \param N array's (uncommon) dimension
     * \param K both arrays' common dimensions
     **/
    template <typename Ty_>
    __host__ std::vector<Ty_> matmul_cuda(const Ty_* a, const Ty_* b, uint32_t M, uint32_t N, uint32_t K);


    /**
     * \brief Performs matrix multiplication using cuBLAS on two matrices
     * \param a pointer to a 1D serialized array with size of M*K
     * \param b pointer to a 1D serialized array with size of N*K
     * \param M array's (uncommon) dimension
     * \param N array's (uncommon) dimension
     * \param K both arrays' common dimensions
     **/
    template <typename Ty_>
    __host__ std::vector<Ty_> matmul_cublas(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K);

    /**
     * \brief Performs a test on the correctness of the matrix multiplication implementations
     * \param dim dimension size of the sample vectors
     **/
    template <typename Ty_>
    void test_matrix_multiplication_correctness(uint32_t dim);

};