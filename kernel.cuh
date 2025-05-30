#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <type_traits>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include "benchmark.h"
#include "eigen/Eigen/Dense"

using std::cout;
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
__global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, unsigned int M, unsigned int K, unsigned int N);

// HOST FUNCTIONS

// param:
// int device (GPU ID)
__host__ void CUDAContextInit();

template<typename Ty_>
std::vector<Ty_> matmul_flat(const std::vector<Ty_>& A, const std::vector<Ty_>& B, unsigned int M, unsigned int K, unsigned int N);

template<typename Ty_>
std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, unsigned int M, unsigned int K, unsigned int N);

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction);

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction);

template <typename Ty_>
__host__ std::vector<Ty_> matrixMul(const std::vector<Ty_>& a, const std::vector<Ty_>& b, unsigned int M, unsigned int K, unsigned int N);