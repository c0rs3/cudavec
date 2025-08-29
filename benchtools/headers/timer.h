#ifndef BENCHMARK_TIMER
#define BENCHMARK_TIMER

#include <iostream>
#include <chrono>
#include <ctime>
#include <time.h>
#include <sstream>
#include <iomanip>

namespace benchtools {
    using duration = std::chrono::duration<double>;
    
    /******************************************************************************
     * @brief Enum for setting the time unit for benchtools::Timer
     ******************************************************************************/
    enum timeunit {
        nanosecond = 0x3B9ACA00,
        microsecond = 0x000003E8,
        milisecond = 0x000F4240,
        second = 0x00000000,
        Default = 0x00000001,
        minute = 0x0000003C,
        hour = 0x00000E10,
    };
    
    std::chrono::duration<double> durationCast(std::chrono::duration<double>& otherDuration, timeunit unit);

    /******************************************************************************
     * @brief Stores the duration from the last deconstructor called
     * @warning DO NOT TRY TO EDIT THE VARIABLE
     * @note This variable is not thread-safe exactly.
     * @note Depending on the time of access another duration from an another .cpp file might be returned
 ******************************************************************************/
    static std::chrono::duration<double> LAST_DURATION;

    /******************************************************************************
    * @brief Timer class
    * @note define EXPLICIT_TIMER to explicit the destructor and manually call the destructor
    * @note define EXPLICIT_LOG to prevent logging of duration to the ostream
     ******************************************************************************/
    class Timer {
    private:
        std::chrono::duration<double> mDuration;
        std::chrono::steady_clock::time_point mStart;
        timeunit mUnit = Default;
    public:
        Timer();

        Timer(const timeunit& unit);
#ifndef EXPLICIT_TIMER
        /******************************************************************************
        * @brief Destructor of the timer class
        * @note define EXPLICIT_TIMER to explicit the destructor and manually call the destructor
         ******************************************************************************/
        ~Timer();
#else
        /******************************************************************************
        * @brief Explicit destructor of the timer class
        * @note undefine EXPLICIT_TIMER to implicit the destructor
         ******************************************************************************/
        ~Timer();
#endif
        void SetUnit(const timeunit& unit);
    };
}

std::string return_current_time_and_date();
#endif
