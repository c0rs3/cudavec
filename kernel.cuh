#pragma once

// CUDA 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUBLAS
#include <cublas_v2.h>
#include <cublasLt.h>

// STL
#include <type_traits>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <assert.h>
#include <windows.h>
#include <fstream>

#include "benchmark.hpp"

#define BENCHMARK 1
#define BENCHMARK_FLAT 0

using std::cout;
using std::clog;
using std::endl;
using std::cerr;
using std::flush;

template<typename Ty_>
__global__ void addKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size);

template<typename Ty_>
__global__ void mulKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size);

template<typename Ty_>
__global__ void divKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size);

template<typename Ty_>
__global__ void addEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size);

template<typename Ty_>
__global__ void mulEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size);

template<typename Ty_>
__global__ void divEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size);

template <typename Ty_>
__global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, unsigned int M, unsigned int N, unsigned int K);

/*
* \brief Lazy loading for CUDA kernel calls
* \param device - The GPU (device) id
*/
__host__ void CUDAContextInit(int device);

template<typename Ty_>
__host__ std::vector<Ty_> matmul_flat(const Ty_* A, const Ty_* B, unsigned int M, unsigned int N, unsigned int K);

template<typename Ty_>
__host__ std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, unsigned int M, unsigned int N, unsigned int K);

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction);

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction);

/**
 * \brief Matrix multiplication function which uses regular CUDA
 *
 * Note that this function uses page-locked memory (host memory).
 *
 * \param A  -  First matrix
 * \param B  -  Second Matrix
 * \param M  -  First matrix column size
 * \param N  -  Second matrix column size
 * \param K  -  First & Second matrix row size
 *
 * \return
 * std::vector<Ty_>
 */
template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda(const Ty_* a, const Ty_* b, unsigned int M, unsigned int N, unsigned int K);

/**
 * \brief Matrix multiplication function which uses CUBLAS API
 * \param A  -  First matrix
 * \param B  -  Second Matrix
 * \param M  -  First matrix column size
 * \param N  -  Second matrix column size
 * \param K  -  First & Second matrix row size
 *
 * \return
 * std::vector<Ty_>
 */
template <typename Ty_>
__host__ std::vector<Ty_> matmul_cublas(const Ty_* A, const Ty_* B, unsigned int M, unsigned int N, unsigned int K);