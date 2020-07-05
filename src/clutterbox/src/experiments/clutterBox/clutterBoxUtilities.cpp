
#include <cuda_runtime_api.h>
#include "clutterBoxUtilities.h"

SpinImage::gpu::Mesh combineMeshesOnGPU(std::vector<SpinImage::gpu::Mesh> meshes) {
    unsigned int totalVertexCount = 0;
    for(unsigned int i = 0; i < meshes.size(); i++) {
        totalVertexCount += meshes.at(i).vertexCount;
    }

    SpinImage::gpu::Mesh combinedMesh;

    cudaMalloc((void**) &combinedMesh.normals_x, totalVertexCount * sizeof(float));
    cudaMalloc((void**) &combinedMesh.normals_y, totalVertexCount * sizeof(float));
    cudaMalloc((void**) &combinedMesh.normals_z, totalVertexCount * sizeof(float));

    cudaMalloc((void**) &combinedMesh.vertices_x, totalVertexCount * sizeof(float));
    cudaMalloc((void**) &combinedMesh.vertices_y, totalVertexCount * sizeof(float));
    cudaMalloc((void**) &combinedMesh.vertices_z, totalVertexCount * sizeof(float));

    combinedMesh.vertexCount = totalVertexCount;

    size_t pointerBaseIndex = 0;
    for(unsigned int i = 0; i < meshes.size(); i++) {
        size_t meshBufferSize = meshes.at(i).vertexCount * sizeof(float);

        cudaMemcpy(&combinedMesh.normals_x[pointerBaseIndex], meshes.at(i).normals_x, meshBufferSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&combinedMesh.normals_y[pointerBaseIndex], meshes.at(i).normals_y, meshBufferSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&combinedMesh.normals_z[pointerBaseIndex], meshes.at(i).normals_z, meshBufferSize, cudaMemcpyDeviceToDevice);

        cudaMemcpy(&combinedMesh.vertices_x[pointerBaseIndex], meshes.at(i).vertices_x, meshBufferSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&combinedMesh.vertices_y[pointerBaseIndex], meshes.at(i).vertices_y, meshBufferSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&combinedMesh.vertices_z[pointerBaseIndex], meshes.at(i).vertices_z, meshBufferSize, cudaMemcpyDeviceToDevice);

        pointerBaseIndex += meshes.at(i).vertexCount;
    }

    return combinedMesh;
}