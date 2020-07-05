#pragma once

#include <vector>
#include <string>
#include <random>

std::vector<std::string> generateRandomFileList(const std::string &objectDirectory,
                                                unsigned int sampleSetSize,
                                                std::minstd_rand0 &generator);