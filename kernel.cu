#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "benchmark.h"
#include <iostream>
#include <immintrin.h>

typedef std::vector<std::vector<int>> Matrix;

using std::cout;
using std::endl;
using std::cerr;
using std::flush;


// KERNELS

__global__ static void KernelWarmup() {
}

template<typename Ty_>
__global__ static void addKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

template<typename Ty_>
__global__ static void mulKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b[i];
	}
}

template<typename Ty_>
__global__ static void divKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b[i];
	}
}

template<typename Ty_>
__global__ static void addEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b;
	}
}

template<typename Ty_>
__global__ static void mulEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b;
	}
}

template<typename Ty_>
__global__ static void divEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b;
	}
}

template <typename Ty_>
__global__ static void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, unsigned int M, unsigned int K, unsigned int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		Ty_ sum = 0;
		for (int i = 0; i < K; ++i) {
			sum += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}
}

// HOST FUNCTIONS

__host__ void CUDAContextInit(int device = 0) {
	// cudastatus for tracking errors
	cudaError_t cudaStatus = cudaSuccess;

	// Set device (GPU)
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return;
	}

	KernelWarmup << <1, 1 >> > ();
	cudaDeviceSynchronize();
}

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction);

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction);

template <typename Ty_>
__host__ std::vector<Ty_> matrixMul(const std::vector<Ty_>& a, const std::vector<Ty_>& b, unsigned int M, unsigned int K, unsigned int N);

template<typename Ty_>
std::vector<Ty_> operator+(const std::vector<Ty_>& left, const std::vector<Ty_>& right) {
	return performOperator(left, right, addKernel);
}

template<typename Ty_>
std::vector<Ty_> operator+(const std::vector<Ty_>& left, const Ty_& right) {
	return performOperator(left, right, addKernel);
}

template<typename Ty_>
std::vector<Ty_> operator*(const std::vector<Ty_>& left, const std::vector<Ty_>& right) {
	return performOperator(left, right, addKernel);
}

template<typename Ty_>
std::vector<Ty_> matmul_flat(const std::vector<Ty_>& A, const std::vector<Ty_>& B, unsigned int M, unsigned int K, unsigned int N) {
	std::vector<Ty_> C(M * N, Ty_(0));

	for (unsigned int i = 0; i < M; ++i) {
		for (unsigned int k = 0; k < K; ++k) {
			Ty_ a_ik = A[i * K + k];
			for (unsigned int j = 0; j < N; ++j) {
				C[i * N + j] += a_ik * B[k * N + j];
			}
		}
	}

	return C;
}

template<typename Ty_>
std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, unsigned int M, unsigned int K, unsigned int N) {
	Ty_* C = new Ty_[M * N];
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; j += 8) { // 8 floats per AVX register
			__m256 c_vec = _mm256_setzero_ps();

			for (int k = 0; k < K; ++k) {
				__m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
				__m256 a_val = _mm256_set1_ps(A[i * K + k]);
				c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
			}

			_mm256_storeu_ps(&C[i * N + j], c_vec);
		}
	}
	return std::vector<Ty_>(C, C + M * N);
}

int main() {
	CUDAContextInit();
	for (size_t i = 0; i < 10; i++) {

	}
	const size_t size = 1 << 16;
	const size_t dim = 1 << 8;
	std::vector<float> A(size);
	std::vector<float> B(size);

	for (size_t i = 0; i < size; ++i) {
		A[i] = i;
		B[i] = i;
	}

	{
		std::clog << "AVX:\n";
		benchmark::Timer<float> timer;
		matmul_avx(A.data(), B.data(), dim, dim, dim);
	}
	std::chrono::duration<float> dur1 = benchmark::dur;
	// std::clog << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(benchmark::dur).count() << "ms" << endl;

	{
		std::clog << "CUDA:\n";
		benchmark::Timer<float> timer;
		matrixMul(A, B, dim, dim, dim);
	}
	std::chrono::duration<float> dur2 = benchmark::dur;
	if (!dur1.count() > dur2.count()) {
		std::cout << "stop\n";
		break;
	}
	// std::clog << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(benchmark::dur).count() << "ms" << endl;
	return 0;
}

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction) {
	CUDAContextInit();
	// cudastatus for tracking errors
	cudaError_t cudaStatus = cudaSuccess;

	// Set device (GPU)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return {};
	}

	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr;

	// Vector size
	size_t size = a.size() > b.size() ? b.size() : a.size();

	// Pinned memory pointer
	Ty_* c;

	// CUDA stream
	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);


	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}

	// Allocate pinned host memory
	cudaMallocHost(&c, size * sizeof(Ty_));

	// Allocate device memory
	cudaMalloc(&dev_a, size * sizeof(Ty_));
	cudaMalloc(&dev_b, size * sizeof(Ty_));

	// Copy data from host to device asynchronously
	cudaStatus = cudaMemcpyAsync(dev_a, a.data(), size * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	cudaStatus = cudaMemcpyAsync(dev_b, b.data(), size * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	// Kernel launch configuration
	dim3 blocksPerGrid(1024);
	dim3 threadsPerBlock(size / 1024);
	kernelFunction << <blocksPerGrid, threadsPerBlock, 0, stream >> > (c, dev_a, dev_b, size);

	// Synchronize the stream to ensure all tasks are complete
	cudaStatus = cudaStreamSynchronize(stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to synchronize streams!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	std::vector<Ty_> res(c, c + size);

	// Cleanup
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFreeHost(c);
	cudaStatus = cudaStreamDestroy(stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to destroy stream!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	return res;
}

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction) {
	// cudastatus for tracking errors
	cudaError_t cudaStatus = cudaSuccess;

	// Set device (GPU)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return {};
	}

	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr;

	// Vector size
	size_t size = a.size();

	// Pinned memory pointer
	Ty_* c;

	// CUDA stream
	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}

	// Allocate pinned host memory
	cudaMallocHost(&c, size * sizeof(Ty_));

	// Allocate device memory
	cudaMalloc(&dev_a, size * sizeof(Ty_));
	cudaMalloc(&dev_b, sizeof(Ty_));

	// Copy data from host to device asynchronously
	cudaStatus = cudaMemcpyAsync(dev_a, a.data(), size * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}
	cudaStatus = cudaMemcpyAsync(dev_b, &b, sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	// Kernel launch configuration
	dim3 blocksPerGrid(1024);
	dim3 threadsPerBlock(size / 1024);
	kernelFunction << <blocksPerGrid, threadsPerBlock, 0, stream >> > (c, dev_a, dev_b, size);

	// Synchronize the stream to ensure all tasks are complete
	cudaStatus = cudaStreamSynchronize(stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to synchronize streams!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	std::vector<Ty_> res(c, c + size);

	// Cleanup
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFreeHost(c);
	cudaStatus = cudaStreamDestroy(stream);

	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to destroy stream!" << std::endl;
		return {};
	}

	return res;
}

template <typename Ty_>
__host__ std::vector<Ty_> matrixMul(const std::vector<Ty_>& a, const std::vector<Ty_>& b, unsigned int M, unsigned int K, unsigned int N) {
	// cudastatus for tracking errors
	cudaError_t cudaStatus = cudaSuccess;

	// Set device (GPU)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return {};
	}

	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr;

	// Vector size
	size_t size_a = M * K;
	size_t size_b = K * N;

	// Pinned memory pointer
	Ty_* c;

	// CUDA stream
	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}

	// Allocate pinned host memory
	cudaMallocHost(&c, M * N * sizeof(Ty_));

	// Allocate device memory
	cudaMalloc(&dev_a, size_a * sizeof(Ty_));
	cudaMalloc(&dev_b, size_b * sizeof(Ty_));

	// Copy data from host to device asynchronously
	cudaStatus = cudaMemcpyAsync(dev_a, a.data(), size_a * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	cudaStatus = cudaMemcpyAsync(dev_b, b.data(), size_b * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	// Kernel launch configuration
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
	matmul_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (dev_a, dev_b, c, M, K, N);

	// Synchronize the stream to ensure all tasks are complete
	cudaStatus = cudaStreamSynchronize(stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to synchronize streams!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	std::vector<Ty_> res(c, c + M * N);

	// Cleanup
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFreeHost(c);
	cudaStatus = cudaStreamDestroy(stream);

	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to destroy stream!" << std::endl;
		return {};
	}

	return res;
}