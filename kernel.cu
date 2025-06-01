#include "kernel.cuh" 
// KERNELS

// Empty kernel call for context initialization
__global__ void KernelWarmup() {}

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
__global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, unsigned int M, unsigned int K, unsigned int N) {
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
std::vector<Ty_> matmul_flat(const std::vector<Ty_>& A, const std::vector<Ty_>& B, unsigned int M, unsigned int K, unsigned int N) {
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
std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, unsigned int M, unsigned int K, unsigned int N) {
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

void bench(size_t start_dim = 1, size_t end_dim = 16) {

	for (size_t k = start_dim; k <= end_dim; k++) {
		const size_t size = 1 << k * 2;
		const size_t dim = 1 << k;

		std::vector<float> A(size);
		std::vector<float> B(size);
		for (size_t i = 0; i < size; ++i) {
			A[i] = i;
			B[i] = i;
		}
		std::clog << "Element size:" << (1 << k * 2) << " || Dimensions: " << dim << "x" << dim << endl;
		
		{
			std::clog << "AVX:" << endl;
			benchmark::Timer<float> timer1;
			matmul_avx(A.data(), B.data(), dim, dim, dim);
		}		
		std::clog << endl;

		{
			std::clog << "CUDA:" << endl;
			benchmark::Timer<float> timer2;
			matrixMul(A, B, dim, dim, dim);
		}
		std::clog << endl;

		{
			std::clog << "CPU:" << endl;
			benchmark::Timer<float> timer4;
			matmul_flat(A, B, dim, dim, dim);
		}
		std::clog << endl;


	}
}

template <typename Ty_>
void test_matrix_multiplication_correctness(size_t dim) {
	const size_t size = dim * dim;
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
	std::vector<Ty_> res_flat = matmul_flat(A, B, dim, dim, dim);
	std::vector<Ty_> res_cuda = matrixMul(A, B, dim, dim, dim);

	auto check_equal = [&](const std::vector<Ty_>& computed, const std::vector<Ty_>& reference, const std::string& label) {
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

	auto check_equal_f = [&](const std::vector<float>& computed, const std::vector<Ty_>& reference, const std::string& label) {
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

	if (std::is_same<Ty_, float>::value) {
		check_equal_f(res_avx, res_flat, "AVX");
	}
	check_equal(res_cuda, res_flat, "CUDA");

	std::clog << "All implementations passed correctness test for size " << dim << "x" << dim << ".\n";
}



int main() {
	CUDAContextInit();
	// test_matrix_multiplication_correctness<float>(static_cast<size_t>(1) << 10);
	bench();
	return 0;
}
