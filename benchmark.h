#ifndef _BENCHMARK_
#define _BENCHMARK_
#include <iostream>
#include <chrono>
#define _BENCHMARK_BEGIN namespace benchmark {
#define _BENCHMARK_END }

_BENCHMARK_BEGIN

template<typename type>
class Timer {
private:
public:
    static size_t alloc_size;

	std::chrono::duration<type> duration;
    std::chrono::steady_clock::time_point start; 

    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::clog << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms" << " || ";
        std::clog << "Duration: " << std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() << "ns" << std::endl;
    }
};

_BENCHMARK_END

#endif