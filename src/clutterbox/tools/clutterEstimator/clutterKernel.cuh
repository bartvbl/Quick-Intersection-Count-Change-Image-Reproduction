#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <vector>

SpinImage::array<float> computeClutter(
        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> array,
        SpinImage::gpu::PointCloud cloud,
        float spinImageWidth,
        size_t referenceObjectSampleCount,
        size_t referenceObjectOriginCount);
size_t computeReferenceSampleCount(
        SpinImage::gpu::Mesh referenceMesh,
        size_t totalSceneSampleCount,
        SpinImage::array<float> cumulativeAreaArray);