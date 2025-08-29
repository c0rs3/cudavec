#include <cudavec.cuh>

int main() {
	Logger logger{ "log.txt", std::ios::app };
	CUDAContextInit(0);
	size_t sample_size = 1, size_limit = 10;
	for (uint32_t k = 1; k <= size_limit; ++k) {
		const uint32_t size = static_cast<uint32_t>(1) << k * 2;
		const uint32_t dim = static_cast<uint32_t>(1) << k;

		std::vector<float> A(size);
		std::vector<float> B(size);
		for (uint32_t i = 0; i < size; ++i) {
			A[i] = i;
			B[i] = i;
		}
		std::clog << "Testing: " << dim << "x" << dim << ".\n";
		logger.log(std::to_string(size));
		float dur1 = 0, dur2 = 0, dur3 = 0, dur4 = 0;
		for (uint32_t i = 0; i < sample_size; i++) {
			std::vector<float> res1;
			{
				std::clog << "cublas total time:" << endl;
				benchtools::Timer timer;
				res1 = matmul_cublas(A.data(), B.data(), dim, dim, dim);
			}
			dur1 += benchtools::LAST_DURATION.count(); {
				std::clog << "cuda total time:" << endl;
				benchtools::Timer timer;
				res1 = matmul_cuda(A.data(), B.data(), dim, dim, dim);
			}
			dur2 += benchtools::LAST_DURATION.count(); {
				std::clog << "Flat total time:" << endl;
				benchtools::Timer timer;
				res1 = matmul_flat(A.data(), B.data(), dim, dim, dim);
			}
#if OS_WINDOWS
			dur3 += benchtools::LAST_DURATION.count(); {
				std::clog << "AVX total time:" << endl;
				benchtools::Timer timer;
				res1 = matmul_avx(A.data(), B.data(), dim, dim, dim);
#endif
		}
			dur4 += benchtools::LAST_DURATION.count();
	}

		logger.log(std::to_string(dur1 / sample_size));
		logger.log(std::to_string(dur2 / sample_size));
		logger.log(std::to_string(dur3 / sample_size));
		logger.log(std::to_string(dur4 / sample_size));
}
}
