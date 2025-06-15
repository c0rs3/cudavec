# cudaVec
- Implementation of matrix multipication _(and few other operators)_ with CUDA
- All kernel functions are wrapped/can be wrapped in a helper function
- A lazy loading function `CudaContextInit()`, which speeds up the initial kernel call 4x at worst
## Benchmark Results
- Even matrices multiplied
- All matrix multiplication results are correct and can be verified
 ```test_matrix_multiplication_correctness<typename>([control size])```
 - _Note: for floating-point vectors sensitivity is adjusted whilst comparing_

### Specs:
- GPU: Intel I9-14900HX
- GPU: RTX 4060 Mobile
### Configuration
- CUDA Toolkit Version 12.9
- Compiler: MSVC + nvcc
- Launch configuration: Release mode
- ```/O2``` and ```-use_fast_math``` enabled

![graph smh](benchss.png "Title")

- 80x speed up on GPU compared to CPU and 18x compared to AVX Instructions
- Comparable performance with CUBLAS


### Lazy Loading
- With lazy loading (1024 x 1024 matrices)
```cpp
CUDA:
Duration(ms): 7ms
Duration(ns): 7033400ns
CUDA:
Duration(ms): 6ms
Duration(ns): 6937900ns
```
- Without lazy loading (1024 x 1024 matrices)
```cpp
CUDA:
Duration(ms): 77ms
Duration(ns): 77046496ns
CUDA:
Duration(ms): 7ms
Duration(ns): 7368700ns
```
