#include <fstream>

#define OVERWRITE std::ios::out
#define PUSHBACK std::ios::app

class Logger {
private:
	std::string file_name;
	std::ofstream stream;
public:
	Logger(std::string file_name, std::ios::openmode mode);

	~Logger();

	void log(const std::string& arg);

	void log(const double& arg);

	void clear();
};