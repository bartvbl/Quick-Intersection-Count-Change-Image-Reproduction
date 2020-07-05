#include "listDir.h"
#include <experimental/filesystem>

std::vector<std::string> listDir(std::string directory) {
	std::vector<std::string> directoryContents;

	for (const auto & entry : std::experimental::filesystem::directory_iterator(directory)) {
		std::string fileName = entry.path().filename().string();
		if(fileName != "." && fileName != "..") {
			directoryContents.push_back(fileName);
		}
	}

    return directoryContents;
}