#pragma once

#include <vector>
#include <string>
#include <experiments/clutterBoxExperiment.hpp>
#include <experimental/filesystem>

enum class BenchmarkMode {
    BASELINE,
    SPHERE_CLUTTER
};

void runQuicciDistanceFunctionBenchmark(
        std::experimental::filesystem::path sourceDirectory,
        std::experimental::filesystem::path outputDirectory,
        size_t seed,
        std::vector<int> sphereCountList,
        int sceneSphereCount,
        float clutterSphereRadius,
        GPUMetaData gpuMetaData,
        float supportRadius,
        BenchmarkMode mode,
        bool dumpSceneOBJ);