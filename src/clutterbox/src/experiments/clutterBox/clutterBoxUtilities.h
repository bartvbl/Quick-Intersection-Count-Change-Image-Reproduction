#pragma once

#include <vector>
#include <spinImage/gpu/types/Mesh.h>

SpinImage::gpu::Mesh combineMeshesOnGPU(std::vector<SpinImage::gpu::Mesh> meshes);