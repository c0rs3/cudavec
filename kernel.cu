#include "utils.h"
#include <iostream>

__global__ static void addKernel(int* c, const int* a, const int* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

__global__ static void mulKernel(int* c, const int* a, const int* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b[i];
	}
}

__global__ static void divKernel(int* c, const int* a, const int* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b[i];
	}
}

__global__ static void addEqualsKernel(int* c, const int* a, const int& b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b;
	}
}

__global__ static void mulEqualsKernel(int* c, const int* a, const int& b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] * b;
	}
}

__global__ static void divEqualsKernel(int* c, const int* a, const int& b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] / b;
	}
}


template<typename Ty_>
std::vector<Ty_> operator+(const std::vector<Ty_>& left, const std::vector<Ty_>& right) {
	std::vector<Ty_> res;
	res.resize(left.size());
	for (size_t i = 0; i < left.size(); i++) {
		res[i] = left[i] + right[i];
	}
	return res;
}

template<typename Ty_>
std::vector<Ty_> operator*(const std::vector<Ty_>& left, const std::vector<Ty_>& right) {
	std::vector<Ty_> res;
	res.resize(left.size());
	for (size_t i = 0; i < left.size(); i++) {
		res[i] = left[i] * right[i];
	}
	return res;
}


int main() {
	const size_t size = 1 << 20;
	std::vector<int> A(size);
	std::vector<int> B(size);

	for (size_t i = 0; i < size; ++i) {
		A[i] = i;
		B[i] = i;
	}
	{
		benchmark::Timer<float> Timer;
		std::vector<int> res = A + B;

	}
	{
		benchmark::Timer<float> Timer;
		std::vector<int> res = performOperator(A, B, addKernel);

	}

	std::cout << std::endl;

	return 0;
}
