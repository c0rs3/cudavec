#include <cudavec.cuh>

__global__ void KernelWarmup() {}

template<typename Ty_>
__global__ void addKernel(Ty_* c, const Ty_* a, const Ty_* b, uint32_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

template<typename Ty_>
__global__ void mulKernel(Ty_* c, const Ty_* a, const Ty_* b, uint32_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b[i];
	}
}

template<typename Ty_>
__global__ void divKernel(Ty_* c, const Ty_* a, const Ty_* b, uint32_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b[i];
	}
}

template<typename Ty_>
__global__ void addEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, uint32_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b;
	}
}

template<typename Ty_>
__global__ void mulEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, uint32_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b;
	}
}

template<typename Ty_>
__global__ void divEqualsKernel(Ty_* c, const Ty_* a, const Ty_& b, uint32_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b;
	}
}

template <typename Ty_>
__global__ void matmul_kernel(const Ty_* A, const Ty_* B, Ty_* C, uint32_t M, uint32_t N, uint32_t K) {
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
	std::clog << "Initializing context for device: " << device << std::endl;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return;
	}

	cudaDeviceProp deviceProps;
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to retrieve device properties! (incompatible GPU?)" << std::endl;
		return;
	}

	KernelWarmup << <1, 1 >> > ();
	cudaDeviceSynchronize();

	cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
	std::clog << deviceProps << std::endl;
}

__host__ std::ostream& operator<<(std::ostream& stream, const cudaDeviceProp& devProps) {
	stream
		<< "Device Properties:\n"
		<< "Device name: " << devProps.name << "\n"
		<< "totalGlobalMem: " << (devProps.totalGlobalMem) << " bytes" << "\n"
		<< "sharedMemPerBlock: " << devProps.sharedMemPerBlock << " bytes" << "\n"
		// << "regsPerBlock: " << devProps.regsPerBlock << "\n"
		<< "maxThreadsPerBlock: " << devProps.maxThreadsPerBlock << " threads" << "\n"
		<< "maxThreadsDim(x): " << devProps.maxThreadsDim[0] << " threads" << "\n"
		// << "maxThreadsDim(y): " << devProps.maxThreadsDim[1] << " threads" << "\n"
		// << "maxThreadsDim(z): " << devProps.maxThreadsDim[2] << " threads" << "\n"
		<< "maxGridSize(x): " << devProps.maxGridSize[0] << " grids" << "\n"
		// << "maxGridSize(y): " << devProps.maxGridSize[1] << " grids" << "\n"
		// << "maxGridSize(z): " << devProps.maxGridSize[2] << " grids" << "\n"
		<< "major CUDA compute capability: " << devProps.major << "\n"
		<< "minor CUDA compute capability: " << devProps.minor << "\n"
		<< "multiProcessorCount: " << devProps.multiProcessorCount << " processors" << "\n"
		<< "memoryBusWidth: " << devProps.memoryBusWidth << " bits" << "\n"
		// << "l2CacheSize: " << (devProps.l2CacheSize / 1000000) << "MB" << "\n"
		<< "maxThreadsPerMultiProcessor: " << devProps.maxThreadsPerMultiProcessor << " threads" << "\n";
	return stream;
}

DeviceMemoryStatus::DeviceMemoryStatus() {
	cudaError_t cudaStatus;

	cudaStatus = cudaMemGetInfo(&mFreeAmount, &mTotalAmount);
	if (cudaStatus) {
		std::clog << "Failed to retrieve memory info from the current context device\n" << std::flush;
	}
	mUsedAmount = mTotalAmount - mFreeAmount;
}

__host__ std::ostream& operator<<(std::ostream& stream, const DeviceMemoryStatus& memStatus) {
	stream
		<< "Free amount: " << (memStatus.mFreeAmount / (1024 * 1024)) << "MB" << std::endl
		<< "Used amount: " << (memStatus.mUsedAmount / (1024 * 1024)) << "MB" << std::endl
		<< "Total available amount: " << (memStatus.mTotalAmount / (1024 * 1024)) << "MB" << std::flush;
	return stream;
}


template<typename Ty_>
__host__ std::vector<Ty_> matmul_flat(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K) {
	std::vector<Ty_> C(M * N, 0);

	for (uint32_t i = 0; i < M; i++) {
		for (uint32_t k = 0; k < K; k++) {
			Ty_ a_ik = A[i * K + k];
			for (uint32_t j = 0; j < N; j++) {
				C[i * N + j] += a_ik * B[k * N + j];
			}
		}
	}

	return C;
}

#if OS_WINDOWS
template<typename Ty_>
__host__ std::vector<Ty_> matmul_avx(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K) {
	Ty_* C = new Ty_[M * N];
	std::memset(C, 0, sizeof(Ty_) * M * N);

	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = 0; j < N; j += 8) {
			__m256 c_vec = _mm256_setzero_ps();

			for (uint32_t k = 0; k < K; ++k) {
				__m256 b_vec;
				if (j + 8 <= N) {
					b_vec = _mm256_loadu_ps(&B[k * N + j]);
				}
				else { // Tail handling 
					float tmp[8] = {};
					for (uint32_t t = 0; t < N - j; ++t)
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
				for (uint32_t t = 0; t < N - j; ++t)
					C[i * N + j + t] = tmp[t];
			}
		}
	}

	std::vector<Ty_> result(C, C + M * N);
	delete[] C;
	return result;
}
#endif

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction) {
	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return {};
	}

	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr;

	// Operation size
	size_t size = a.size() > b.size() ? b.size() : a.size();

	// Pinned memory pointer
	Ty_* c = nullptr;

	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);

	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}

	cudaMallocHost(&c, size * sizeof(Ty_));
	cudaMalloc(&dev_a, size * sizeof(Ty_));
	cudaMalloc(&dev_b, size * sizeof(Ty_));

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
	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return {};
	}

	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr;

	// Operation size
	size_t size = a.size();

	// Pinned memory pointer
	Ty_* c = nullptr;

	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}

	cudaMallocHost(&c, size * sizeof(Ty_));
	cudaMalloc(&dev_a, size * sizeof(Ty_));
	cudaMalloc(&dev_b, sizeof(Ty_));

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
__host__ std::vector<Ty_> matmul_cuda(const Ty_* a, const Ty_* b, uint32_t M, uint32_t N, uint32_t K) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaDeviceProp deviceProps;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
		return {};
	}

	cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to retrieve device properties! (incompatible GPU?)" << std::endl;
		return {};
	}

	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr;

	// Vector sizes
	size_t size_a = M * K;
	size_t size_b = K * N;

	// Pinned memory pointer
	Ty_* c = nullptr;

	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}


	cudaMallocAsync(&dev_a, size_a * sizeof(Ty_), stream);
	cudaMallocAsync(&dev_b, size_b * sizeof(Ty_), stream);

	cudaMallocHost(&c, M * N * sizeof(Ty_));

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
	// TODO: Better launch configuration
	uint32_t threadsPerBlock = deviceProps.maxThreadsPerBlock;
	uint32_t blocksPerGrid = threadsPerBlock;
	matmul_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (dev_a, dev_b, c, M, N, K);

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
__host__ std::vector<Ty_> matmul_cublas(const Ty_* A, const Ty_* B, uint32_t M, uint32_t N, uint32_t K) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed to create stream!" << std::endl;
		cudaStreamDestroy(stream);
		return {};
	}

	Ty_* dev_a = nullptr, * dev_b = nullptr, * dev_c = nullptr;
	cudaMallocAsync(&dev_a, M * K * sizeof(Ty_), stream);
	cudaMallocAsync(&dev_b, K * N * sizeof(Ty_), stream);

	cudaMallocAsync(&dev_c, M * N * sizeof(Ty_), stream);

	cudaStatus = cudaMemcpyAsync(dev_a, A, M * K * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(dev_c);
		cudaStreamDestroy(stream);

		return {};
	}
	cudaStatus = cudaMemcpyAsync(dev_b, B, K * N * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "Failed memcpy!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFreeHost(dev_c);
		cudaStreamDestroy(stream);

		return {};
	}

	cublasHandle_t handle;
	cublasCreate(&handle);

	Ty_ alpha = 1.0, beta = 0.0;

	if (std::is_same<Ty_, float>::value) {
		cublasSgemm_v2(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, K,
			&alpha,
			dev_a, M,
			dev_b, K,
			&beta,
			dev_c, M
		);
	}
	else {
		cublasSgemm_v2(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, K,
			&alpha,
			dev_a, M,
			dev_b, K,
			&beta,
			dev_c, M
		);
	}
	std::vector<Ty_> res = std::vector<Ty_>(M * N);
	cudaMemcpyAsync(res.data(), dev_c, M * N * sizeof(Ty_), cudaMemcpyDeviceToHost, stream);
	cudaDeviceSynchronize();

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaStreamDestroy(stream);
	cublasDestroy(handle);

	return res;
}

template <typename Ty_>
void test_matrix_multiplication_correctness(uint32_t dim) {
	const uint32_t size = dim * dim;
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
#if OS_WINDOWS
		res_avx = matmul_avx(Af.data(), Bf.data(), dim, dim, dim);
#endif
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
				return;
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
				return;
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

template std::vector<float> matmul_cublas<float>(const float*, const float*, uint32_t, uint32_t, uint32_t);
template std::vector<float> matmul_cuda<float>(const float*, const float*, uint32_t, uint32_t, uint32_t);
template std::vector<float> matmul_flat<float>(const float*, const float*, uint32_t, uint32_t, uint32_t);
#if OS_WINDOWS
template std::vector<float> matmul_avx<float>(const float*, const float*, uint32_t, uint32_t, uint32_t);
#endif