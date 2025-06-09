#include "kernel.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
// KERNELS

// Empty kernel call for context initialization
__global__ static void KernelWarmup() {}

template<typename Ty_>
__global__ void addKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

template<typename Ty_>
__global__ void mulKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b[i];
	}
}

template<typename Ty_>
__global__ void divKernel(Ty_* c, const Ty_* a, const Ty_* b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b[i];
	}
}

template<typename Ty_>
__global__ void addEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b;
	}
}

template<typename Ty_>
__global__ void mulEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b;
	}
}

template<typename Ty_>
__global__ void divEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b;
	}
}

template <typename Ty_>
__global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, unsigned int M, unsigned int N, unsigned int K) {
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

__host__ void CUDAContextInit(int device = 0) {
	cudaError_t cudaStatus = cudaSuccess;

	// Set device (GPU)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return;
	}

	KernelWarmup << <1, 1 >> > ();
	cudaDeviceSynchronize();
}

template<typename Ty_>
__host__ std::vector<Ty_> matmul_flat(const Ty_* A, const Ty_* B, unsigned int M, unsigned int N, unsigned int K) {
	std::vector<Ty_> C(M * N, 0);

	for (unsigned int i = 0; i < M; i++) {
		for (unsigned int k = 0; k < K; k++) {
			Ty_ a_ik = A[i * K + k];
			for (unsigned int j = 0; j < N; j++) {
				C[i * N + j] += a_ik * B[k * N + j];
			}
		}
	}

	return C;
}

template<typename Ty_>
__host__ std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, unsigned int M, unsigned int N, unsigned int K) {
	Ty_* C = new Ty_[M * N];
	std::memset(C, 0, sizeof(Ty_) * M * N);

	for (unsigned int i = 0; i < M; ++i) {
		for (unsigned int j = 0; j < N; j += 8) {
			__m256 c_vec = _mm256_setzero_ps();

			for (unsigned int k = 0; k < K; ++k) {
				__m256 b_vec;
				if (j + 8 <= N) {
					b_vec = _mm256_loadu_ps(&B[k * N + j]);
				}
				else {
					// Tail handling
					float tmp[8] = {};
					for (unsigned int t = 0; t < N - j; ++t)
						tmp[t] = B[k * N + j + t];
					b_vec = _mm256_loadu_ps(tmp);
				}

				__m256 a_val = _mm256_set1_ps(A[i * K + k]);
				c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
			}

			if (j + 8 <= N) {
				_mm256_storeu_ps(&C[i * N + j], c_vec);
			}
			else {
				float tmp[8];
				_mm256_storeu_ps(tmp, c_vec);
				for (unsigned int t = 0; t < N - j; ++t)
					C[i * N + j + t] = tmp[t];
			}
		}
	}

	std::vector<Ty_> result(C, C + M * N);
	delete[] C;
	return result;
}

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction) {
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
	Ty_* c = nullptr;

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
	Ty_* c = nullptr;

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
__host__ std::vector<Ty_> matmul_cuda(const Ty_* a, const Ty_* b, unsigned int M, unsigned int N, unsigned int K) {
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
	Ty_* c = nullptr;

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

	cudaStatus = cudaMemcpyAsync(dev_a, a, size_a * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(c);
		cudaStreamDestroy(stream);

		return {};
	}

	cudaStatus = cudaMemcpyAsync(dev_b, b, size_b * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
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

	matmul_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (dev_a, dev_b, c, M, N, K);


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
	std::vector<Ty_> res = std::vector<Ty_>(c, c + M * N);

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
__host__ std::vector<Ty_> matmul_cublas(const Ty_* A, const Ty_* B, unsigned int M, unsigned int N, unsigned int K) {
	// static_assert(std::is_same<Ty_, float>::value || std::is_same<Ty_, double>::value, "Ty_ must be float or double");
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	Ty_* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
	cudaMallocAsync(&d_A, M * K * sizeof(Ty_), stream);
	cudaMallocAsync(&d_B, K * N * sizeof(Ty_), stream);

	cudaMallocAsync(&d_C, M * N * sizeof(Ty_), stream);

	cudaMemcpyAsync(d_A, A, M * K * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B, B, K * N * sizeof(Ty_), cudaMemcpyHostToDevice, stream);

	cublasHandle_t handle;
	cublasCreate(&handle);

	Ty_ alpha = 1.0, beta = 0.0;

	if constexpr (std::is_same<Ty_, float>::value) {
		cublasSgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, K,
			&alpha,
			d_A, M,  
			d_B, K,  
			&beta,
			d_C, M
		);
	}
	else {
		cublasDgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, K,
			&alpha,
			d_A, M,
			d_B, K,
			&beta,
			d_C, M
		);
	}
	std::vector<Ty_> res = std::vector<Ty_>(M * N);
	cudaMemcpyAsync(res.data(), d_C, M * N * sizeof(Ty_), cudaMemcpyDeviceToHost, stream);
	cudaDeviceSynchronize();

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaStreamDestroy(stream);
	cublasDestroy(handle);

	return res;
}

template <typename Ty_>
void test_matrix_multiplication_correctness(unsigned int dim) {
	const unsigned int size = dim * dim;
	std::vector<Ty_> A(size), B(size);

	for (size_t i = 0; i < size; ++i) {
		A[i] = static_cast<Ty_>(i);
		B[i] = static_cast<Ty_>(i);
	}

	// All implementation results
	std::vector<float> res_avx;
	if (std::is_same<Ty_, float>::value) {
		std::vector<float> Af(size), Bf(size);

		for (size_t i = 0; i < size; ++i) {
			Af[i] = static_cast<float>(i);
			Bf[i] = static_cast<float>(i);
		}
		res_avx = matmul_avx(Af.data(), Bf.data(), dim, dim, dim);
	}

	std::vector<Ty_> res_flat = matmul_flat(A.data(), B.data(), dim, dim, dim);
	std::vector<Ty_> res_cuda = matmul_cuda(A.data(), B.data(), dim, dim, dim);
	std::vector<Ty_> res_clas = matmul_cublas(A.data(), B.data(), dim, dim, dim);

	auto check_equal = [&size](const std::vector<Ty_>& computed, const std::vector<Ty_>& reference, const std::string& label) {
		std::cout << "check_equal for " << label << std::endl;
		for (size_t i = 0; i < size; ++i) {
			if (i % 100 == 0)
				std::cout << "\rLines remaining: " << size - i << " " << std::flush;

			if (std::abs(computed[i] - reference[i]) > (pow(10, log10(computed[i]) - 1))) {
				std::cerr << std::endl << label << " mismatch at index " << i
					<< ": got " << computed[i]
					<< ", expected " << reference[i] << std::endl;
				assert(false);
			}
		}
		std::cout << "\rLines remaining: " << 0 << " " << std::flush << std::endl;

		};

	auto check_equal_f = [&size](const std::vector<float>& computed, const std::vector<Ty_>& reference, const std::string& label) {
		std::cout << "check_equal for " << label << std::endl;
		for (size_t i = 0; i < size; ++i) {
			if (i % 100 == 0)
				std::cout << "\rLines remaining: " << (size - i) - ((size - i) % 100) << " " << std::flush;

			if (std::abs(computed[i] - reference[i]) > (pow(10, log10(computed[i]) - 1))) {
				std::cerr << std::endl << label << " mismatch at index " << i
					<< ": got " << computed[i]
					<< ", expected " << reference[i] << std::endl;
				assert(false);
			}
		}
		std::cout << "\rLines remaining: " << 0 << " " << std::flush << std::endl;
		};

	if (std::is_same<Ty_, float>::value) {
		check_equal_f(res_avx, res_flat, "AVX");
	}
	check_equal(res_cuda, res_flat, "CUDA");
	check_equal(res_clas, res_flat, "CLAS");

	std::clog << "All implementations passed correctness test for size " << dim << "x" << dim << ".\n";
}

int main() {
	CUDAContextInit();

	/*
	const size_t size = 1 << 10 * 2;
	const size_t dim = 1 << 10;

	std::vector<float> A(size);
	std::vector<float> B(size);
	for (size_t i = 0; i < size; ++i) {
		A[i] = i;
		B[i] = i;
	}
	std::clog << "Testing " << dim << "x" << dim << ".\n";
	std::vector<float> res1, res2, res3;
	{
		std::clog << "Flat total time:" << endl;
		benchmark::Timer<float> timer;
		res1 = matmul_flat(A.data(), B.data(), dim, dim, dim);
	}
	{
		std::clog << "cublas total time:" << endl;
		benchmark::Timer<float> timer;
		res2 = matmul_cublas(A.data(), B.data(), dim, dim, dim);
	}
	{
		std::clog << "cuda total time:" << endl;
		benchmark::Timer<float> timer;
		res3 = matmul_cuda(A.data(), B.data(), dim, dim, dim);
	}
	*/
	test_matrix_multiplication_correctness<float>(2);
	return 0;
}
