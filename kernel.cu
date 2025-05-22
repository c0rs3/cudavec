#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <functional>
#include <vector>
#include "benchmark.h"
#include <iostream>

typedef std::vector<std::vector<int>> Matrix2x2;

using std::cout;
using std::endl;
using std::cerr;
using std::flush;

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

// TODO add more operators
__global__ void matrix_multiplication_kernel(int* A, int* B, int* C, unsigned int M, unsigned int N, unsigned int K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t k = 0; k < N; k++) {

		}
	}
}

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
	int threadsPerBlock = 1024;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
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
__host__ std::vector<Ty_> matrixMul(const std::vector<Ty_>&a, const Ty_ & b) {
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
	int threadsPerBlock = 1024;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
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



// to multiply two matrices
Matrix2x2 mult(const Matrix2x2& arr, const Matrix2x2& brr) {
	int n = arr.size();

	// to store the resultant matrix
	Matrix2x2 res(n, std::vector<int>(n, 0));

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				res[i][j] += arr[i][k] * brr[k][j];
			}
		}
	}

	return res;
}

int main() {
	CUDAContextInit();
	const size_t size = 2;
	std::vector<std::vector<int>> A(size);
	A[0] = { 1, 2 };
	A[1] = { 3, 4 };
	std::vector<std::vector<int>> B(size);
	B[0] = { 5, 6 };
	B[1] = { 7, 8 };
	std::vector<std::vector<int>> C(size);
	C[0] = { 0, 0 };
	C[1] = { 0, 0 };

	std::vector<std::vector<int>> res = mult(A, B);
	for (int i = 0; i < res.size(); i++) {
		for (int j = 0; j < res[i].size(); j++) {
			cout << res[i][j] << " ";
		}
		cout << endl;
	}
	// matrix_multiplication_kernel(A.data()->data(), B.data()->data(), C.data()->data(), A.size(), A[0].size(), B.size());
#ifdef PERF_DEBUG
	const size_t size = 1 << 20;
	std::vector<int> A(size);
	std::vector<int> B(size);

	for (size_t i = 0; i < size; ++i) {
		A[i] = i;
		B[i] = i;
	}

	{
		benchmark::Timer<float> timer;
		performOperator(A, B, addKernel<int>);
	}
#endif
	return 0;
}
