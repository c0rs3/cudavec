#include <fstream>

#define OVERWRITE std::ios::out
#define PUSHBACK std::ios::app

class Logger {
private:
	std::string file_name;
	std::ofstream stream;
public:
	/******************************************************************************
	* @brief Constructor for the logger object.
	* @param file_name file name
	* @param fileMode file open mode
	* @note use std::ios::openmode enum to specify the mode
	 ******************************************************************************/
	Logger(std::string file_name, std::ios::openmode mode);

	/******************************************************************************
	* @brief Destructor for the logger object which logs the log end time to the file
	 ******************************************************************************/
	~Logger();

	void log(const std::string& arg);

	void log(const double& arg);

	void clear();
};