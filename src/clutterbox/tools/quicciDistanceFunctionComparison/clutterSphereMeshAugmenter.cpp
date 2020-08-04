#include <vector>
#include <glm/glm.hpp>
#include <random>
#include <algorithm>
#include <iostream>
#include "clutterSphereMeshAugmenter.h"
#include <spinImage/utilities/meshSampler.cuh>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/utilities/copy/DeviceVertexList.h>


void generateSphere(std::vector<SpinImage::cpu::float3> &vertices, std::vector<SpinImage::cpu::float3> &normals, float sphereRadius, int slices, int layers) {
    const unsigned int triangleCount = slices * layers * 2;

    vertices.reserve(3 * triangleCount);
    normals.reserve(3 * triangleCount);

    // Slices require us to define a full revolution worth of triangles.
    // Layers only requires angle varying between the bottom and the top (a layer only covers half a circle worth of angles)
    const float degreesPerLayer = 180.0f / (float) layers;
    const float degreesPerSlice = 360.0f / (float) slices;

    // Constructing the sphere one layer at a time
    for (int layer = 0; layer < layers; layer++) {
        int nextLayer = layer + 1;

        // Angles between the vector pointing to any point on a particular layer and the negative z-axis
        float currentAngleZDegrees = degreesPerLayer * float(layer);
        float nextAngleZDegrees = degreesPerLayer * float(nextLayer);

        // All coordinates within a single layer share z-coordinates.
        // So we can calculate those of the current and subsequent layer here.
        float currentZ = -std::cos(glm::radians(currentAngleZDegrees));
        float nextZ = -std::cos(glm::radians(nextAngleZDegrees));

        // The row of vertices forms a circle around the vertical diagonal (z-axis) of the sphere.
        // These radii are also constant for an entire layer, so we can precalculate them.
        float radius = std::sin(glm::radians(currentAngleZDegrees));
        float nextRadius = std::sin(glm::radians(nextAngleZDegrees));

        // Now we can move on to constructing individual slices within a layer
        for (int slice = 0; slice < slices; slice++) {

            // The direction of the start and the end of the slice in the xy-plane
            float currentSliceAngleDegrees = float(slice) * degreesPerSlice;
            float nextSliceAngleDegrees = float(slice + 1) * degreesPerSlice;

            // Determining the direction vector for both the start and end of the slice
            float currentDirectionX = std::cos(glm::radians(currentSliceAngleDegrees));
            float currentDirectionY = std::sin(glm::radians(currentSliceAngleDegrees));

            float nextDirectionX = std::cos(glm::radians(nextSliceAngleDegrees));
            float nextDirectionY = std::sin(glm::radians(nextSliceAngleDegrees));

            vertices.emplace_back(sphereRadius * radius * currentDirectionX,
                                  sphereRadius * radius * currentDirectionY,
                                  sphereRadius * currentZ);
            vertices.emplace_back(sphereRadius * radius * nextDirectionX,
                                  sphereRadius * radius * nextDirectionY,
                                  sphereRadius * currentZ);
            vertices.emplace_back(sphereRadius * nextRadius * nextDirectionX,
                                  sphereRadius * nextRadius * nextDirectionY,
                                  sphereRadius * nextZ);
            vertices.emplace_back(sphereRadius * radius * currentDirectionX,
                                  sphereRadius * radius * currentDirectionY,
                                  sphereRadius * currentZ);
            vertices.emplace_back(sphereRadius * nextRadius * nextDirectionX,
                                  sphereRadius * nextRadius * nextDirectionY,
                                  sphereRadius * nextZ);
            vertices.emplace_back(sphereRadius * nextRadius * currentDirectionX,
                                  sphereRadius * nextRadius * currentDirectionY,
                                  sphereRadius * nextZ);

            normals.emplace_back(radius * currentDirectionX,
                                 radius * currentDirectionY,
                                 currentZ);
            normals.emplace_back(radius * nextDirectionX,
                                 radius * nextDirectionY,
                                 currentZ);
            normals.emplace_back(nextRadius * nextDirectionX,
                                 nextRadius * nextDirectionY,
                                 nextZ);
            normals.emplace_back(radius * currentDirectionX,
                                 radius * currentDirectionY,
                                 currentZ);
            normals.emplace_back(nextRadius * nextDirectionX,
                                 nextRadius * nextDirectionY,
                                 nextZ);
            normals.emplace_back(nextRadius * currentDirectionX,
                                 nextRadius * currentDirectionY,
                                 nextZ);
        }
    }
}

SpinImage::cpu::Mesh applyClutterSpheres(SpinImage::cpu::Mesh inputMesh, int count, float radius, size_t randomSeed) {
    std::vector<SpinImage::cpu::float3> sphereVertices;
    std::vector<SpinImage::cpu::float3> sphereNormals;
    std::cout << "Input mesh has " << inputMesh.vertexCount << " vertices." << std::endl;

    generateSphere(sphereVertices, sphereNormals, radius, SPHERE_RESOLUTION_X, SPHERE_RESOLUTION_Y);

    size_t combinedVertexCount = inputMesh.vertexCount + count * SPHERE_VERTEX_COUNT;

    // Indices are ignored on the GPU
    SpinImage::cpu::Mesh outputMesh(combinedVertexCount, 0);

    const unsigned int sampleCount = 10000000;

    SpinImage::gpu::Mesh device_mesh = SpinImage::copy::hostMeshToDevice(inputMesh);
    SpinImage::gpu::PointCloud sampledMesh = SpinImage::utilities::sampleMesh(device_mesh, sampleCount, randomSeed);
    SpinImage::gpu::freeMesh(device_mesh);

    SpinImage::array<SpinImage::cpu::float3> sampleVertices = SpinImage::copy::deviceVertexListToHost(sampledMesh.vertices);
    SpinImage::array<SpinImage::cpu::float3> sampleNormals = SpinImage::copy::deviceVertexListToHost(sampledMesh.normals);

    sampledMesh.free();

    // Copy the original mesh
    std::copy(inputMesh.vertices, inputMesh.vertices + inputMesh.vertexCount, outputMesh.vertices);
    std::copy(inputMesh.normals, inputMesh.normals + inputMesh.vertexCount, outputMesh.normals);

    // Select random list of vertices
    std::minstd_rand0 generator{randomSeed};
    std::vector<unsigned int> vertexIndices;
    vertexIndices.resize(sampleCount);
    for(int i = 0; i < sampleCount; i++) {
        vertexIndices.at(i) = i;
    }
    std::shuffle(vertexIndices.begin(), vertexIndices.end(), generator);

    // Place spheres on surface
    for(int i = 0; i < count; i++) {
        SpinImage::cpu::float3 sampleVertex = sampleVertices.content[vertexIndices.at(i)];
        SpinImage::cpu::float3 sampleNormal = sampleNormals.content[vertexIndices.at(i)];

        SpinImage::cpu::float3 sphereOrigin = sampleVertex + radius * sampleNormal;

        unsigned int startIndex = inputMesh.vertexCount + i * SPHERE_VERTEX_COUNT;
        for(int j = 0; j < SPHERE_VERTEX_COUNT; j++) {
            outputMesh.vertices[startIndex + j] = sphereVertices.at(j) + sphereOrigin;
            outputMesh.normals[startIndex + j] = sphereNormals.at(j);
        }
    }

    delete[] sampleVertices.content;
    delete[] sampleNormals.content;

    outputMesh.vertexCount = inputMesh.vertexCount + count * SPHERE_VERTEX_COUNT;

    return outputMesh;
}