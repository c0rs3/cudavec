#pragma once

// CUDA 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
template <typename Ty_>
__global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, uint32_t M, uint32_t N, uint32_t K);
__host__ void CUDAContextInit(int device);
template<typename Ty_>
__host__ std::vector<Ty_> matmul_flat(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K);
#if OS_WINDOWS
template<typename Ty_>
__host__ std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K);
#elif OS_LINUX
#else
#endif
template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction);
template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction);
template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda(const Ty_* a, const Ty_* b, uint32_t M, uint32_t N, uint32_t K);
template <typename Ty_>
__host__ std::vector<Ty_> matmul_cublas(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K);

template <typename Ty_>
void test_matrix_multiplication_correctness(uint32_t dim);