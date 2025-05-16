#ifndef CUDAVEC_H
#define CUDAVEC_H
#include<vector>

template <typename Ty_, typename KernelFunc>
std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const std::vector<Ty_>& b, KernelFunc kernelFunction) {
	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr, * dev_c = nullptr;

	size_t size = a.size();
	Ty_* c;

	// CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate pinned host memory
	cudaMallocHost(&c, size * sizeof(Ty_));

	// Allocate device memory
	cudaMalloc(&dev_a, size * sizeof(Ty_));
	cudaMalloc(&dev_b, size * sizeof(Ty_));

	// Copy data from host to device asynchronously
	cudaMemcpyAsync(dev_a, a.data(), size * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dev_b, b.data(), size * sizeof(Ty_), cudaMemcpyHostToDevice, stream);

	// Kernel launch configuration
	int threadsPerBlock = 1024;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	kernelFunction << <blocksPerGrid, threadsPerBlock, 0, stream >> > (c, dev_a, dev_b, size);

	// Synchronize the stream to ensure all tasks are complete
	cudaStreamSynchronize(stream);

	std::vector<Ty_> res(c, c + size);

	// Cleanup
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(c);
	cudaStreamDestroy(stream);

	return res;
}

template <typename Ty_, typename KernelFunc>
std::vector<Ty_> performOperator(const std::vector<Ty_>& a, const Ty_& b, KernelFunc kernelFunction) {
	// Device pointers
	Ty_* dev_a = nullptr, * dev_b = nullptr, * dev_c = nullptr;

	size_t size = a.size();
	Ty_* c;

	// CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate pinned host memory
	cudaMallocHost(&c, size * sizeof(Ty_));

	// Allocate device memory
	cudaMalloc(&dev_a, size * sizeof(Ty_));
	cudaMalloc(&dev_b, size * sizeof(Ty_));
	cudaMalloc(&dev_c, size * sizeof(Ty_));

	// Copy data from host to device asynchronously
	cudaMemcpyAsync(dev_a, a.data(), size * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(dev_b, &b, sizeof(Ty_), cudaMemcpyHostToDevice, stream);

	// Kernel launch configuration
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	kernelFunction << <blocksPerGrid, threadsPerBlock, 0, stream >> > (dev_c, dev_a, dev_b, size);

	cudaMemcpyAsync(c, dev_c, size * sizeof(Ty_), cudaMemcpyDeviceToHost, stream);

	// Synchronize the stream to ensure all tasks are complete
	cudaStreamSynchronize(stream);

	std::vector<Ty_> res(c, c + size);

	// Cleanup
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(c);
	cudaStreamDestroy(stream);

	return res;
}

#endif 
