#include <arrrgh.hpp>
#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/QUICCImages.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <spinImage/utilities/duplicateRemoval.cuh>
#include <spinImage/gpu/radialIntersectionCountImageGenerator.cuh>
#include <spinImage/libraryBuildSettings.h>
#include <cuda_runtime.h>
#include <fstream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/copy/deviceDescriptorsToHost.h>
#include <spinImage/utilities/modelScaler.h>
#include <spinImage/utilities/dumpers/rawDescriptorDumper.h>

const float DEFAULT_SPIN_IMAGE_WIDTH = 0.3;

int main(int argc, const char** argv) {
    arrrgh::parser parser("quiccidumper", "Render QUICCI images from an input OBJ file, and dump them to a compressed (ZIP) file in binary form.");
    const auto& inputOBJFile = parser.add<std::string>("input-obj-file", "Location of the OBJ file from which the images should be rendered.", '\0', arrrgh::Required, "");
    const auto& outputDumpFile = parser.add<std::string>("output-dump-file", "Location where the generated images should be dumped to.", '\0', arrrgh::Required, "");
    const auto& fitInUnitSphere = parser.add<bool>("fit-object-in-unit-sphere", "Scale the object such that it fits in a unit sphere", '\0', arrrgh::Optional, false);
    const auto& spinImageWidth = parser.add<float>("spin-image-width", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_WIDTH);
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    std::cout << "Loading OBJ file: " << inputOBJFile.value() << std::endl;
    SpinImage::cpu::Mesh hostMesh = SpinImage::utilities::loadOBJ(inputOBJFile.value(), true);

    if(fitInUnitSphere.value()) {
        std::cout << "Fitting object in unit sphere.." << std::endl;
        SpinImage::cpu::Mesh scaledMesh = SpinImage::utilities::fitMeshInsideSphereOfRadius(hostMesh, 1);
        SpinImage::cpu::freeMesh(hostMesh);
        hostMesh = scaledMesh;
    }

    SpinImage::gpu::Mesh deviceMesh = SpinImage::copy::hostMeshToDevice(hostMesh);
    SpinImage::cpu::freeMesh(hostMesh);

    std::cout << "Computing QUICCI images.." << std::endl;
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> uniqueVertices =
            SpinImage::utilities::computeUniqueVertices(deviceMesh);

    SpinImage::gpu::QUICCIImages images = SpinImage::gpu::generateQUICCImages(deviceMesh, uniqueVertices, spinImageWidth.value());
    SpinImage::cpu::QUICCIImages hostImages = SpinImage::copy::QUICCIDescriptorsToHost(images);

    std::cout << "Writing output file.." << std::endl,
    SpinImage::dump::raw::descriptors(outputDumpFile.value(), hostImages);

    SpinImage::gpu::freeMesh(deviceMesh);
    cudaFree(uniqueVertices.content);
    cudaFree(images.images);

}

