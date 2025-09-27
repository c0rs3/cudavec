#include <filelog.h>
#include <timer.h>

/******************************************************************************
* @brief Constructor for the logger object.
* @param file_name file name
* @param fileMode file open mode
* @note use std::ios::openmode enum to specify the mode
 ******************************************************************************/
Logger::Logger(std::string file_name, std::ios::openmode fileMode = std::ios::app)
    :file_name(file_name), stream(file_name, fileMode) {
    if (!stream.is_open()) {
        std::clog << "Failed to open a stream to the file!\n" << std::flush;
        std::string exceptionMessage = std::string("Failed to open file") + file_name;
        throw std::runtime_error(exceptionMessage.c_str());
    }
    std::clog << "File open mode: " << fileMode << std::endl;
    stream << "Log start at: " << return_current_time_and_date() << "\n" << std::flush;

}


/******************************************************************************
* @brief Destructor for the logger object which logs the log end time to the file
 ******************************************************************************/
Logger::~Logger() {
    stream << "Log end at: " << return_current_time_and_date() << "\n\n" << std::flush;
}

void Logger::log(const double& arg) {
    stream << arg << "\n" << std::flush;
}

void Logger::log(const std::string& arg) {
    stream << arg << "\n" << std::flush;
}

/******************************************************************************
* @brief Clears the contents of the file
* @warning RAII might be distrupted with this
 ******************************************************************************/
void Logger::clear() {
    stream.close();
    stream.open(file_name, std::ios::trunc);
    stream.flush();
}
