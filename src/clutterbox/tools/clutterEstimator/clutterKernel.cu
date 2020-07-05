#include "clutterKernel.cuh"
#include <nvidia/helper_cuda.h>
#include <nvidia/helper_math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_EQUIVALENCE_ROUNDING_ERROR 0.0001

__device__ __inline__ float3 transformCoordinate(const float3 &vertex, const float3 &spinImageVertex, const float3 &spinImageNormal)
{
    const float2 sineCosineAlpha = normalize(make_float2(spinImageNormal.x, spinImageNormal.y));

    const bool is_n_a_not_zero = !((abs(spinImageNormal.x) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.y) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_ax = is_n_a_not_zero ? sineCosineAlpha.x : 1;
    const float alignmentProjection_n_ay = is_n_a_not_zero ? sineCosineAlpha.y : 0;

    float3 transformedCoordinate = vertex - spinImageVertex;

    const float initialTransformedX = transformedCoordinate.x;
    transformedCoordinate.x = alignmentProjection_n_ax * transformedCoordinate.x + alignmentProjection_n_ay * transformedCoordinate.y;
    transformedCoordinate.y = -alignmentProjection_n_ay * initialTransformedX + alignmentProjection_n_ax * transformedCoordinate.y;

    const float transformedNormalX = alignmentProjection_n_ax * spinImageNormal.x + alignmentProjection_n_ay * spinImageNormal.y;

    const float2 sineCosineBeta = normalize(make_float2(transformedNormalX, spinImageNormal.z));

    const bool is_n_b_not_zero = !((abs(transformedNormalX) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.z) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_bx = is_n_b_not_zero ? sineCosineBeta.x : 1;
    const float alignmentProjection_n_bz = is_n_b_not_zero ? sineCosineBeta.y : 0; // discrepancy between axis here is because we are using a 2D vector on 3D axis.

    // Order matters here
    const float initialTransformedX_2 = transformedCoordinate.x;
    transformedCoordinate.x = alignmentProjection_n_bz * transformedCoordinate.x - alignmentProjection_n_bx * transformedCoordinate.z;
    transformedCoordinate.z = alignmentProjection_n_bx * initialTransformedX_2 + alignmentProjection_n_bz * transformedCoordinate.z;

    return transformedCoordinate;
}

__global__ void computeClutterKernel(
        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> origins,
        SpinImage::gpu::PointCloud samplePointCloud,
        SpinImage::array<float> clutterValues,
        float spinImageWidth,
        size_t referenceObjectSampleCount) {

    __shared__ unsigned long long int hitObjectSampleCount;
    __shared__ unsigned long long int hitClutterSampleCount;

    const size_t pointCloudSampleCount = samplePointCloud.vertices.length;

    if(threadIdx.x == 0) {
        hitObjectSampleCount = 0;
        hitClutterSampleCount = 0;
    }

    unsigned long long int threadObjectSampleCount = 0;
    unsigned long long int threadClutterSampleCount = 0;

    __syncthreads();

    const SpinImage::gpu::DeviceOrientedPoint origin = origins.content[blockIdx.x];
    const float3 spinVertex = origin.vertex;
    const float3 spinNormal = origin.normal;

    for(size_t sampleIndex = threadIdx.x; sampleIndex < pointCloudSampleCount; sampleIndex += blockDim.x) {
        const float3 samplePoint = samplePointCloud.vertices.at(sampleIndex);

        const float3 projectedSampleLocation = transformCoordinate(samplePoint, spinVertex, spinNormal);

        const float alpha = length(make_float2(projectedSampleLocation.x, projectedSampleLocation.y));
        const float beta = projectedSampleLocation.z;

        // if projected point lies inside spin image
        if(alpha <= spinImageWidth && abs(beta) <= (spinImageWidth / 2.0f)) {

            // Determine if sample is clutter
            if(sampleIndex >= referenceObjectSampleCount) {
                threadClutterSampleCount++;
            } else {
                threadObjectSampleCount++;
            }
        }
    }

    // Safely add up all tallies
    atomicAdd(&hitObjectSampleCount, threadObjectSampleCount);
    atomicAdd(&hitClutterSampleCount, threadClutterSampleCount);

    __syncthreads();

    if(threadIdx.x == 0) {
        double clutterPercentage = double(hitObjectSampleCount) / double(hitObjectSampleCount + hitClutterSampleCount);

        if(clutterPercentage == 0) {
            printf("%i\n", blockIdx.x);
        }

        clutterValues.content[blockIdx.x] = float(clutterPercentage);
    }
}

SpinImage::array<float> computeClutter(SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> origins, SpinImage::gpu::PointCloud samplePointCloud, float spinImageWidth, size_t referenceObjectSampleCount, size_t referenceObjectOriginCount) {
    SpinImage::array<float> device_clutterValues;

    size_t clutterBufferSize = referenceObjectOriginCount * sizeof(float);
    checkCudaErrors(cudaMalloc(&device_clutterValues.content, clutterBufferSize));

    cudaMemset(device_clutterValues.content, 0, clutterBufferSize);

    computeClutterKernel<<<referenceObjectOriginCount, 128>>>(origins, samplePointCloud, device_clutterValues, spinImageWidth, referenceObjectSampleCount);
    checkCudaErrors(cudaDeviceSynchronize());

    SpinImage::array<float> host_clutterValues;
    host_clutterValues.content = new float[referenceObjectOriginCount];
    host_clutterValues.length = referenceObjectOriginCount;

    checkCudaErrors(cudaMemcpy(host_clutterValues.content, device_clutterValues.content, clutterBufferSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_clutterValues.content));

    return host_clutterValues;

}

__global__ void computeSampleCount(size_t baseVertexIndex, size_t totalSceneSampleCount, SpinImage::array<float> cumulativeAreaArray, size_t* outputIndex) {

    size_t triangleIndex = baseVertexIndex / 3;

    float maxArea = cumulativeAreaArray.content[cumulativeAreaArray.length - 1];
    float areaStepSize = maxArea / (float)totalSceneSampleCount;

    float areaEnd = cumulativeAreaArray.content[triangleIndex];

    size_t lastIndexInRange = (size_t) (areaEnd / areaStepSize);

    *outputIndex = lastIndexInRange;
}

size_t computeReferenceSampleCount(SpinImage::gpu::Mesh referenceMesh, size_t totalSceneSampleCount, SpinImage::array<float> cumulativeAreaArray) {
    size_t* device_sampleCount;
    checkCudaErrors(cudaMalloc(&device_sampleCount, sizeof(size_t)));

    computeSampleCount<<<1, 1>>>(referenceMesh.vertexCount, totalSceneSampleCount, cumulativeAreaArray, device_sampleCount);
    checkCudaErrors(cudaDeviceSynchronize());

    size_t host_sampleCount = 0;
    checkCudaErrors(cudaMemcpy(&host_sampleCount, device_sampleCount, sizeof(size_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_sampleCount));

    return host_sampleCount;
}