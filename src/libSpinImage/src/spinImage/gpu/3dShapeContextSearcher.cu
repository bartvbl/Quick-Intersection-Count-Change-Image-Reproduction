#include <algorithm>
#include <chrono>
#include "3dShapeContextSearcher.cuh"
#include <nvidia/helper_cuda.h>
#include <cfloat>
#include <host_defines.h>
#include <iostream>
#include <vector_types.h>

const size_t elementsPerShapeContextDescriptor =
        SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
        SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
        SHAPE_CONTEXT_LAYER_COUNT;


__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float warpAllReduceMin(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val = min(__shfl_xor_sync(0xFFFFFFFF, val, mask), val);
    return val;
}

__device__ float compute3DSCPairDistanceGPU(
        shapeContextBinType* needleDescriptor,
        shapeContextBinType* haystackDescriptor,
        float* sharedSquaredSums) {

#define sliceOffset threadIdx.y
    float threadSquaredDistance = 0;
    for(short binIndex = threadIdx.x; binIndex < elementsPerShapeContextDescriptor; binIndex += blockDim.x) {
        float needleBinValue = needleDescriptor[binIndex];
        short haystackBinIndex =
            (binIndex + (sliceOffset * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT));
        // Simple modulo that I think is less expensive
        if(haystackBinIndex >= elementsPerShapeContextDescriptor) {
            haystackBinIndex -= elementsPerShapeContextDescriptor;
        }
        float haystackBinValue = haystackDescriptor[haystackBinIndex];
        float binDelta = needleBinValue - haystackBinValue;
        threadSquaredDistance += binDelta * binDelta;
    }

    float combinedSquaredDistance = warpAllReduceSum(threadSquaredDistance);

    if(threadIdx.x == 0) {
        sharedSquaredSums[sliceOffset] = combinedSquaredDistance;
    }

    __syncthreads();

    // An entire warp must participate in the reduction, so we give the excess threads
    // the highest possible value so that any other value will be lower
    float threadValue = threadIdx.x < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT ?
            sharedSquaredSums[threadIdx.x] : FLT_MAX;
    float lowestDistance = std::sqrt(warpAllReduceMin(threadValue));

    // Some threads will race ahead to the next image pair. Need to avoid that.
    __syncthreads();

    return lowestDistance;
}

__global__ void computeShapeContextSearchResultIndices(
        shapeContextBinType* needleDescriptors,
        shapeContextBinType* haystackDescriptors,
        size_t haystackDescriptorCount,
        float haystackScaleFactor,
        unsigned int* searchResults) {
#define needleDescriptorIndex blockIdx.x

    // Since memory is reused a lot, we cache both the needle and haystack image in shared memory
    // Combined this is is approximately (at default settings) the size of a spin or RICI image

    __shared__ shapeContextBinType referenceDescriptor[elementsPerShapeContextDescriptor];
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        referenceDescriptor[index] = needleDescriptors[elementsPerShapeContextDescriptor * needleDescriptorIndex + index];
    }

    __shared__ shapeContextBinType haystackDescriptor[elementsPerShapeContextDescriptor];
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        haystackDescriptor[index] =
                haystackDescriptors[elementsPerShapeContextDescriptor * needleDescriptorIndex + index]
                * (1.0f/haystackScaleFactor);
    }

    __shared__ float squaredSums[SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT];

    __syncthreads();

    float referenceDistance = compute3DSCPairDistanceGPU(
            referenceDescriptor,
            haystackDescriptor,
            squaredSums);

    // No image pair can have a better distance than 0, so we can just stop the search right here
    if(referenceDistance == 0) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackDescriptorIndex = 0; haystackDescriptorIndex < haystackDescriptorCount; haystackDescriptorIndex++) {
        if(needleDescriptorIndex == haystackDescriptorIndex) {
            continue;
        }

        for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
            haystackDescriptor[index] =
                    haystackDescriptors[elementsPerShapeContextDescriptor * haystackDescriptorIndex + index]
                    * (1.0f/haystackScaleFactor);
        }

        __syncthreads();

        float distance = compute3DSCPairDistanceGPU(
                referenceDescriptor,
                haystackDescriptor,
                squaredSums);

        // We've found a result that's better than the reference one. That means this search result would end up
        // above ours in the search result list. We therefore move our search result down by 1.
        if(distance < referenceDistance) {
            searchResultRank++;
        }
    }

    if(threadIdx.x == 0) {
        searchResults[needleDescriptorIndex] = searchResultRank;
    }
}



SpinImage::array<unsigned int> SpinImage::gpu::compute3DSCSearchResultRanks(
        array<shapeContextBinType> device_needleDescriptors,
        size_t needleDescriptorCount,
        size_t needleDescriptorSampleCount,
        array<shapeContextBinType> device_haystackDescriptors,
        size_t haystackDescriptorCount,
        size_t haystackDescriptorSampleCount,
        SpinImage::debug::SCSearchRunInfo* runInfo) {
    static_assert(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT <= 32);

    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = needleDescriptorCount * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    float haystackScaleFactor = float(double(needleDescriptorSampleCount) / double(haystackDescriptorSampleCount));
    std::cout << "\t\tHaystack scale factor: " << haystackScaleFactor << std::endl;

    auto searchStart = std::chrono::steady_clock::now();

    dim3 blockDimensions = {
        32, SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT, 1
    };
    computeShapeContextSearchResultIndices<<<needleDescriptorCount, blockDimensions>>>(
        device_needleDescriptors.content,
        device_haystackDescriptors.content,
        haystackDescriptorCount,
        haystackScaleFactor,
        device_searchResults);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart);

    array<unsigned int> resultIndices;
    resultIndices.content = new unsigned int[needleDescriptorCount];
    resultIndices.length = needleDescriptorCount;

    checkCudaErrors(cudaMemcpy(resultIndices.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    std::chrono::milliseconds executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - executionStart);

    if(runInfo != nullptr) {
        runInfo->searchExecutionTimeSeconds = double(searchDuration.count()) / 1000.0;
        runInfo->totalExecutionTimeSeconds = double(executionDuration.count()) / 1000.0;
    }

    return resultIndices;
}