#include <iostream>
#include <algorithm>
#include <utilities/listDir.h>
#include "randomFileSelector.h"
#include "stringUtils.h"


std::vector<std::string> generateRandomFileList(const std::string &objectDirectory, unsigned int sampleSetSize,
												std::minstd_rand0 &generator) {

    std::vector<std::string> filePaths(sampleSetSize);

    std::cout << "\tListing object directory..";
    std::vector<std::string> fileList = listDir(objectDirectory);
    std::cout << " (found " << fileList.size() << " files)" << std::endl;

    // Sort the file list to avoid the effects of operating systems ordering files inconsistently.
    std::sort(fileList.begin(), fileList.end());

    std::shuffle(std::begin(fileList), std::end(fileList), generator);

    for (unsigned int i = 0; i < sampleSetSize; i++) {
        filePaths[i] = objectDirectory + (endsWith(objectDirectory, "/") ? "" : "/") + fileList.at(i);
        std::cout << "\t\tSample " << i << ": " << filePaths.at(i) << std::endl;
    }

    return filePaths;
}