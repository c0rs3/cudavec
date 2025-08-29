#include <timer.h>

namespace benchtools {
    Timer::Timer() {
        mStart = std::chrono::steady_clock::now();
    }

    Timer::Timer(const timeunit& unit = timeunit::nanosecond) {
        mStart = std::chrono::steady_clock::now();
        mUnit = unit;
    };
#ifndef EXPLICIT_TIMER
    Timer::~Timer() {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        mDuration = end - mStart;
        LAST_DURATION = std::chrono::duration_cast<std::chrono::milliseconds>(mDuration);
#ifndef EXPLICIT_LOG
        switch (mUnit) {
        case timeunit::nanosecond:
            std::clog << "Duration(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(mDuration).count() << "ns" << std::endl;
            break;
        case timeunit::microsecond:
            std::clog << "Duration(μs): " << std::chrono::duration_cast<std::chrono::microseconds>(mDuration).count() << "μs" << std::endl;
            break;
        case timeunit::milisecond:
            std::clog << "Duration(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(mDuration).count() << "ms" << std::endl;
            break;
        case timeunit::second:
            std::clog << "Duration(seconds): " << std::chrono::duration_cast<std::chrono::seconds>(mDuration).count() << "s" << std::endl;
            break;
        case timeunit::minute:
            std::clog << "Duration(minutes): " << std::chrono::duration_cast<std::chrono::seconds>(mDuration).count() << "m" << std::endl;
            break;
        case timeunit::hour:
            std::clog << "Duration(hours): " << std::chrono::duration_cast<std::chrono::hours>(mDuration).count() << "h" << std::endl;
            break;
        case timeunit::Default:
            std::clog << "Duration(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(mDuration).count() << "ns" << std::endl;
            std::clog << "Duration(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(mDuration).count() << "ms" << std::endl;
            break;
        default:
            break;
        }
#endif
    }
#else
    explicit Timer::~Timer() {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        mDuration = end - mStart;
        last_duration = mDuration;
        switch (mUnit) {
        case timeunit::nanosecond:
            std::clog << "Duration(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(mDuration).count() << "ns" << std::endl;
            break;
        case timeunit::microsecond:
            std::clog << "Duration(μs): " << std::chrono::duration_cast<std::chrono::microseconds>(mDuration).count() << "μs" << std::endl;
            break;
        case timeunit::milisecond:
            std::clog << "Duration(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(mDuration).count() << "ms" << std::endl;
            break;
        case timeunit::second:
            std::clog << "Duration(seconds): " << std::chrono::duration_cast<std::chrono::seconds>(mDuration).count() << "s" << std::endl;
            break;
        case timeunit::minute:
            std::clog << "Duration(minutes): " << std::chrono::duration_cast<std::chrono::seconds>(mDuration).count() << "m" << std::endl;
            break;
        case timeunit::hour:
            std::clog << "Duration(hours): " << std::chrono::duration_cast<std::chrono::hours>(mDuration).count() << "h" << std::endl;
            break;
        case timeunit::Default:
            std::clog << "Duration(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(mDuration).count() << "ns" << std::endl;
            std::clog << "Duration(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(mDuration).count() << "ms" << std::endl;
            break;
        default:
            break;
        }
    }
#endif
    void Timer::SetUnit(const timeunit& unit) {
        mUnit = unit;
    };

    std::chrono::duration<double> durationCast(std::chrono::duration<double>& otherDuration, timeunit unit) {
        if (unit == nanosecond) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(otherDuration);
        }
        else if (unit == nanosecond) {
            return std::chrono::duration_cast<std::chrono::microseconds>(otherDuration);
        }
        else if (unit == nanosecond) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(otherDuration);
        }
        else if (unit == nanosecond) {
            return std::chrono::duration_cast<std::chrono::seconds>(otherDuration);
        }
        else if (unit == nanosecond) {
            return std::chrono::duration_cast<std::chrono::minutes>(otherDuration);
        }
        else if (unit == nanosecond) {
            return std::chrono::duration_cast<std::chrono::hours>(otherDuration);
        }
        else {
            return std::chrono::duration_cast<std::chrono::milliseconds>(otherDuration);
        }
        return std::chrono::duration_cast<std::chrono::milliseconds>(otherDuration);
    };
}


std::string return_current_time_and_date() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    struct tm time_info;
#if defined(__GNUC__)
#define COMPILER_GCC
    localtime_r(&in_time_t, &time_info);
#elif defined(_MSC_VER)
    localtime_s(&time_info, &in_time_t);
#endif
    std::stringstream ss;
    ss << std::put_time(&time_info, "%Y-%m-%d %X");
    return ss.str();
}