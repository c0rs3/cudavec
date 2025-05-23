#ifndef _BENCHMARK_
#define _BENCHMARK_
#include <iostream>
#include <chrono>
#define _BENCHMARK_BEGIN namespace benchmark {
#define _BENCHMARK_END }

_BENCHMARK_BEGIN
std::chrono::duration<float> dur;

template<typename Ty_>
class Timer {
private:
public:
	std::chrono::duration<Ty_> duration;
    std::chrono::steady_clock::time_point start; 
    
    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        dur = duration;
        std::clog << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms" << " || ";
        if (!std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() > 0) {
            std::clog << "Duration: " << std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() << "ns" << std::endl;
        }
    }
};

_BENCHMARK_END

#endif