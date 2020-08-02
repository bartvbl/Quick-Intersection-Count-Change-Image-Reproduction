#pragma once

#include <string>
#include <vector>

struct GPUMetaData {
    std::string name;
    int clockRate;
    size_t memorySizeMB;
};

void runClutterBoxExperiment(
        std::string objectDirectory,
        std::vector<std::string> descriptorList,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float pointDensityRadius3dsc,
        float minSupportRadius3dsc,
        float supportRadius,
        float spinImageSupportAngleDegrees,
        unsigned int fpfhBinCount,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        bool dumpSceneOBJFiles,
        std::string sceneOBJFileDumpDir,
        bool enableMatchVisualisation,
        std::string matchVisualisationOutputDir,
        std::vector<std::string> matchVisualisationDescriptorList,
        unsigned int matchVisualisationThreshold,
        GPUMetaData gpuMetaData,
        size_t overrideSeed = 0);
