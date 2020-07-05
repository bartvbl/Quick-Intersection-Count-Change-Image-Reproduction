#pragma once
#include "spinImage/common/types/array.h"
#include <host_defines.h>

namespace SpinImage {
    namespace gpu {
        struct Mesh {
            float* vertices_x;
            float* vertices_y;
            float* vertices_z;

            float* normals_x;
            float* normals_y;
            float* normals_z;

            size_t vertexCount;

            __host__ __device__ Mesh() {
                vertexCount = 0;
            }
        };

        Mesh duplicateMesh(Mesh mesh);
        void freeMesh(Mesh mesh);


    }
}
