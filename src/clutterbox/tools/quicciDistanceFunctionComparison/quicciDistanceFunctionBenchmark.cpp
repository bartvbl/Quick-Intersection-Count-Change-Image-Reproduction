#include "quicciDistanceFunctionBenchmark.h"
#include "clutterSphereMeshAugmenter.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/utilities/modelScaler.h>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/utilities/duplicateRemoval.cuh>
#include <utilities/randomFileSelector.h>
#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <spinImage/utilities/dumpers/meshDumper.h>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <spinImage/gpu/quickIntersectionCountImageSearcher.cuh>
#include <cassert>
#include <utilities/stringUtils.h>
#include <json.hpp>
#include <tsl/ordered_map.h>
#include <spinImage/utilities/copy/deviceMeshToHost.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

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
        bool dumpSceneOBJ) {
    // 1 Seeding the random number generator
    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};

    std::cout << std::endl << "Running experiment initialisation sequence.." << std::endl;

    // 2 Search SHREC directory for files
    // 3 Make a sample set of n sample objects
    unsigned int sampleSetSize = 0;
    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        sampleSetSize = 1;
    } else if(mode == BenchmarkMode::BASELINE) {
        sampleSetSize = 2;
    }
    std::vector<std::string> filePaths = generateRandomFileList(sourceDirectory.string(), sampleSetSize, generator);

    // 4 Load the models in the sample set
    std::cout << "\tLoading sample model.." << std::endl;
    SpinImage::cpu::Mesh sampleMesh = SpinImage::utilities::loadOBJ(filePaths.at(0), true);
    SpinImage::cpu::Mesh otherSampleMesh;
    if(mode == BenchmarkMode::BASELINE) {
        otherSampleMesh = SpinImage::utilities::loadOBJ(filePaths.at(1), true);
    }

    // 5 Scale all models to fit in a 1x1x1 sphere
    std::cout << "\tScaling meshes.." << std::endl;
    SpinImage::cpu::Mesh scaledMesh = SpinImage::utilities::fitMeshInsideSphereOfRadius(sampleMesh, 1);
    SpinImage::cpu::freeMesh(sampleMesh);
    SpinImage::cpu::Mesh scaledOtherSampleMesh;
    if(mode == BenchmarkMode::BASELINE) {
        scaledOtherSampleMesh = SpinImage::utilities::fitMeshInsideSphereOfRadius(otherSampleMesh, 1);
        SpinImage::cpu::freeMesh(otherSampleMesh);
    }

    // 6 Add clutter spheres to the mesh
    std::cout << "\tAugmenting mesh with spheres.." << std::endl;
    SpinImage::cpu::Mesh augmentedHostMesh;
    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        augmentedHostMesh = applyClutterSpheres(scaledMesh, sceneSphereCount, clutterSphereRadius, generator());
    }

    // 6 Copy meshes to GPU
    std::cout << "\tCopying meshes to device.." << std::endl;
    SpinImage::gpu::Mesh unmodifiedMesh = SpinImage::copy::hostMeshToDevice(scaledMesh);
    SpinImage::cpu::freeMesh(scaledMesh);
    SpinImage::gpu::Mesh augmentedMesh;
    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        augmentedMesh = SpinImage::copy::hostMeshToDevice(augmentedHostMesh);
    }
    SpinImage::gpu::Mesh otherSampleUnmodifiedMesh;
    if(mode == BenchmarkMode::BASELINE) {
        otherSampleUnmodifiedMesh = SpinImage::copy::hostMeshToDevice(scaledOtherSampleMesh);
        SpinImage::cpu::freeMesh(scaledOtherSampleMesh);
    }

    // 8 Remove duplicate vertices
    std::cout << "\tRemoving duplicate vertices.." << std::endl;
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> imageOrigins = SpinImage::utilities::computeUniqueVertices(unmodifiedMesh);
    size_t imageCount = imageOrigins.length;
    size_t referenceImageCount = imageCount;
    size_t sampleImageCount = 0;
    std::cout << "\t\tReduced " << unmodifiedMesh.vertexCount << " vertices to " << imageCount << "." << std::endl;
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> baselineOrigins;
    if(mode == BenchmarkMode::BASELINE) {
        baselineOrigins = SpinImage::utilities::computeUniqueVertices(otherSampleUnmodifiedMesh);
        sampleImageCount = baselineOrigins.length;
    }

    std::cout << "\tGenerating reference QUICCI images.." << std::endl;
    SpinImage::gpu::QUICCIImages device_unmodifiedQuiccImages = SpinImage::gpu::generateQUICCImages(
            unmodifiedMesh,
            imageOrigins,
            supportRadius);

    size_t unmodifiedVertexCount = unmodifiedMesh.vertexCount;
    const size_t verticesPerSphere = SPHERE_VERTEX_COUNT;

    std::vector<SpinImage::array<SpinImage::gpu::QUICCIDistances>> measuredDistances;

    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        for (unsigned int sphereCountIndex = 0; sphereCountIndex < sphereCountList.size(); sphereCountIndex++) {
            unsigned int sphereCount = sphereCountList.at(sphereCountIndex);
            augmentedMesh.vertexCount = unmodifiedVertexCount + sphereCount * verticesPerSphere;
            assert(sphereCount <= sceneSphereCount);
            std::cout << "\tComputing distances for a scene with " << sphereCount << " spheres.." << std::endl;
            std::cout << "\t\tGenerating QUICCI images.." << std::endl;
            SpinImage::debug::QUICCIRunInfo runInfo;
            SpinImage::gpu::QUICCIImages device_augmentedQuiccImages = SpinImage::gpu::generateQUICCImages(
                    augmentedMesh,
                    imageOrigins,
                    supportRadius,
                    &runInfo);

            std::cout << "\t\t\tTook " << runInfo.totalExecutionTimeSeconds << " seconds." << std::endl;

            if(dumpSceneOBJ) {
                std::cout << "\t\t\tDumping scene OBJ file.." << std::endl;
                SpinImage::cpu::Mesh dumpMesh = SpinImage::copy::deviceMeshToHost(augmentedMesh);
                SpinImage::dump::mesh(dumpMesh, outputDirectory / ("scene_similar_" + std::to_string(seed) + "_" + std::to_string(sphereCount) + "_spheres.obj"));
                SpinImage::cpu::freeMesh(dumpMesh);
            }

            std::cout << "\t\tComputing QUICCI distances.." << std::endl;
            SpinImage::array<SpinImage::gpu::QUICCIDistances> sampleDistances = SpinImage::gpu::computeQUICCIElementWiseDistances(
                    device_unmodifiedQuiccImages,
                    device_augmentedQuiccImages,
                    imageCount);

            cudaFree(device_augmentedQuiccImages.images);

            measuredDistances.push_back(sampleDistances);
        }
    } else if(mode == BenchmarkMode::BASELINE) {
        imageCount = std::min<unsigned int>(imageCount, baselineOrigins.length);
        baselineOrigins.length = imageCount;

        std::cout << "\tComputing distances.." << std::endl;
        std::cout << "\t\tGenerating QUICCI images.." << std::endl;
        SpinImage::debug::QUICCIRunInfo runInfo;
        SpinImage::gpu::QUICCIImages device_sampleImages = SpinImage::gpu::generateQUICCImages(
                otherSampleUnmodifiedMesh,
                baselineOrigins,
                supportRadius,
                &runInfo);

        std::cout << "\t\t\tTook " << runInfo.totalExecutionTimeSeconds << " seconds." << std::endl;

        if(dumpSceneOBJ) {
            std::cout << "\t\t\tDumping scene OBJ file.." << std::endl;
            SpinImage::cpu::Mesh dumpMesh = SpinImage::copy::deviceMeshToHost(augmentedMesh);
            SpinImage::dump::mesh(dumpMesh, outputDirectory / ("scene_baseline_" + std::to_string(seed) + ".obj"));
            SpinImage::cpu::freeMesh(dumpMesh);
        }

        std::cout << "\t\tComputing QUICCI distances.." << std::endl;
        SpinImage::array<SpinImage::gpu::QUICCIDistances> sampleDistances = SpinImage::gpu::computeQUICCIElementWiseDistances(
                device_unmodifiedQuiccImages,
                device_sampleImages,
                imageCount);

        cudaFree(device_sampleImages.images);

        measuredDistances.push_back(sampleDistances);
    }

    std::string timestring = getCurrentDateTimeString();
    std::string outputFileName = "quicciDistances_" + timestring + "_" + std::to_string(randomSeed) + ".json";
    std::experimental::filesystem::path outputFilePath =  outputDirectory / outputFileName;

    std::cout << std::endl;
    std::cout << "Experiments done, dumping results to file: " << std::endl;
    std::cout << "    " << outputFilePath.string() << std::endl;

    json outJson;

    outJson["version"] = "v2";
    outJson["seed"] = seed;
    outJson["spinImageWidthPixels"] = spinImageWidthPixels;
    outJson["sphereCounts"] = sphereCountList;
    outJson["clutterSphereRadius"] = clutterSphereRadius;
    outJson["sceneSphereCount"] = sceneSphereCount;
    outJson["sphereSliceCount"] = SPHERE_RESOLUTION_X;
    outJson["sphereLayerCount"] = SPHERE_RESOLUTION_Y;
    outJson["trianglesPerSphere"] = SPHERE_TRIANGLE_COUNT;
    outJson["verticesPerSphere"] = SPHERE_VERTEX_COUNT;
    outJson["supportRadius"] = supportRadius;
    outJson["chosenObjectPath"] = filePaths.at(0);
    outJson["imageCount"] = imageCount;
    outJson["objectVertexCount"] = unmodifiedVertexCount;
    outJson["gpuInfo"] = {};
    outJson["gpuInfo"]["name"] = gpuMetaData.name;
    outJson["gpuInfo"]["clockrate"] = gpuMetaData.clockRate;
    outJson["gpuInfo"]["memoryCapacityInMB"] = gpuMetaData.memorySizeMB;
    if(mode == BenchmarkMode::BASELINE) {
        outJson["mode"] = "baseline";
        outJson["primaryObjectImageCount"] = referenceImageCount;
        outJson["secondaryObjectImageCount"] = sampleImageCount;
        outJson["secondaryObjectPath"] = filePaths.at(1);
    } else if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        outJson["mode"] = "similar";
    }

    outJson["measuredDistances"] = {};
    outJson["measuredDistances"]["clutterResistant"] = {};
    outJson["measuredDistances"]["weightedHamming"] = {};
    outJson["measuredDistances"]["hamming"] = {};
    outJson["imageBitCounts"] = {};
    for(unsigned int sphereCountIndex = 0; sphereCountIndex < sphereCountList.size(); sphereCountIndex++) {
        std::string keyName = std::to_string(sphereCountList.at(sphereCountIndex)) + " spheres";
        outJson["measuredDistances"]["clutterResistant"][keyName] = {};
        outJson["measuredDistances"]["weightedHamming"][keyName] = {};
        outJson["measuredDistances"]["hamming"][keyName] = {};

        for(int image = 0; image < imageCount; image++) {
            outJson["measuredDistances"]["clutterResistant"][keyName].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].clutterResistantDistance);
            outJson["measuredDistances"]["weightedHamming"][keyName].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].weightedHammingDistance);
            outJson["measuredDistances"]["hamming"][keyName].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].hammingDistance);
            if(sphereCountIndex == 0) {
                outJson["imageBitCounts"].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].needleImageBitCount);
            }
        }
    }

    std::ofstream outFile(outputFilePath);
    outFile << outJson.dump(4);
    outFile.close();

    std::cout << std::endl << "Complete." << std::endl;
}