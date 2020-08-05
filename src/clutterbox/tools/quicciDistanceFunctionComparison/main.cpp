#include <arrrgh.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvidia/helper_cuda.h"

#include "experiments/clutterBoxExperiment.hpp"
#include "quicciDistanceFunctionBenchmark.h"


#include <stdexcept>

cudaDeviceProp setCurrentCUDADevice(bool listOnly, int forceGPU);

const float DEFAULT_SPIN_IMAGE_WIDTH = 1;
const float DEFAULT_SPIN_IMAGE_SUPPORT_ANGLE_DEGREES = 90;

void splitByCharacter(std::vector<std::string>* parts, const std::string &s, char delim) {

    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        parts->push_back(item);
    }
}

int main(int argc, const char **argv)
{
	arrrgh::parser parser("quicciDistanceFunctionBenchmark", "Benchmarks the different QUICCI distance functions proposed in our paper");
	const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);
	const auto& listGPUs = parser.add<bool>("list-gpus", "List all GPU's, used for the --force-gpu parameter.", 'a', arrrgh::Optional, false);
	const auto& forceGPU = parser.add<int>("force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
	const auto& objectDirectory = parser.add<std::string>("source-directory", "Defines the directory from which input objects are read", '\0', arrrgh::Optional, "");
    const auto& forcedSeed = parser.add<std::string>("force-seed", "Specify the seed to use for random generation. Used for reproducing results.", '\0', arrrgh::Optional, "0");
    const auto& waitOnCompletion = parser.add<bool>("wait-for-input-on-completion", "I needed the program to wait before exiting after completing the experiment. This does that job perfectly. Don't judge.", '\0', arrrgh::Optional, false);
    const auto& supportRadius = parser.add<float>("support-radius", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, 0.3);
	const auto& outputDirectory = parser.add<std::string>("output-directory", "Specify the location where output files should be dumped", '\0', arrrgh::Optional, "../output/");
    const auto& sphereCounts = parser.add<std::string>("sphere-counts", "Specify the number of clutter spheres that should be added into the scene and for which results should be generated, as a comma separated list WITHOUT spaces (e.g. --sphere-counts=1,2,5)", '\0', arrrgh::Optional, "NONE");
    const auto& mode = parser.add<std::string>("experiment-mode", "Determines whether baseline or similar geometry results should be generated. Can be either \"baseline\" or \"similar\"", '\0', arrrgh::Optional, "NOT SPECIFIED");
    const auto& sceneSphereCount = parser.add<int>("scene-sphere-count", "If you want a specified number of objects to be used for the experiment (for ensuring consistency between seeds)", '\0', arrrgh::Optional, -1);
    const auto& clutterSphereRadius = parser.add<float>("clutter-sphere-radius", "Specifies the radius of spheres that should be added into the scene (note: the sample object is first fit inside a unit sphere, so this radius is relative to a unit sphere).", '\0', arrrgh::Optional, 0.05);
    const auto& enableOBJDump = parser.add<bool>("enable-obj-dump", "When enabled, will produce an OBJ file for each tested scene", '\0', arrrgh::Optional, false);


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

	// First, we create a CUDA context on the best compute device.
	// This is naturally the device with most memory available, becausewhywouldntit.

	std::cout << std::endl;
	
	cudaDeviceProp device_information = setCurrentCUDADevice(listGPUs.value(), forceGPU.value());
	GPUMetaData gpuMetaData;
	gpuMetaData.name = std::string(device_information.name);
	gpuMetaData.clockRate = device_information.clockRate;
	gpuMetaData.memorySizeMB = device_information.totalGlobalMem / (1024 * 1024);

	if(listGPUs.value()) {
		return 0;
	}

	if(sphereCounts.value() == "NONE") {
		std::cout << "Experiment requires the --sphere-counts parameter to be set" << std::endl;
		exit(0);
	}

	if(objectDirectory.value().empty()) {
		std::cout << "Experiment requires the --source-directory parameter to be set" << std::endl;
		exit(0);
	}

	if(mode.value() == "NOT SPECIFIED") {
        std::cout << "Experiment requires the --experiment-mode parameter to be set" << std::endl;
        exit(0);
	}

	// Interpret seed value
    std::stringstream sstr(forcedSeed.value());
    size_t randomSeed;
    sstr >> randomSeed;
    if(randomSeed != 0) {
        std::cout << "Using overridden seed: " << randomSeed << std::endl;
    }

    // Interpret the object counts string
    std::vector<std::string> sphereCountParts;
	splitByCharacter(&sphereCountParts, sphereCounts.value(), ',');
    std::vector<int> sphereCountList;
    for (const auto &objectCountPart : sphereCountParts) {
        sphereCountList.push_back(std::stoi(objectCountPart));
    }

    BenchmarkMode benchmarkMode;
    if(mode.value() == "baseline") {
        benchmarkMode = BenchmarkMode::BASELINE;
    } else if (mode.value() == "similar") {
        benchmarkMode = BenchmarkMode::SPHERE_CLUTTER;
    } else {
        std::cout << "The mode specified by the --experiment-mode parameter was set to \"" + mode.value() + "\", but was not recognised. Valid values for this parameter are \"baseline\" and \"similar\"." << std::endl;
        exit(0);
    }

    std::sort(sphereCountList.begin(), sphereCountList.end());

    if(benchmarkMode == BenchmarkMode::BASELINE) {
        sphereCountList = {0};
    }


    // Run benchmark
    runQuicciDistanceFunctionBenchmark(
            objectDirectory.value(),
            outputDirectory.value(),
            randomSeed,
            sphereCountList,
            sceneSphereCount.value(),
            clutterSphereRadius.value(),
            gpuMetaData,
            supportRadius.value(),
            benchmarkMode,
            enableOBJDump.value());


    if(waitOnCompletion.value()) {
        std::cout << "Experiment complete, press enter to exit" << std::endl;
        std::cin.ignore();
    }

    return 0;
}

cudaDeviceProp setCurrentCUDADevice(bool listOnly, int forceGPU)
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	if(listOnly) {	
		std::cout << "Found " << deviceCount << " devices:" << std::endl;
	}

	size_t maxAvailableMemory = 0;
	cudaDeviceProp deviceWithMostMemory;
	int chosenDeviceIndex = 0;
	
	for(int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp deviceProperties;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, i));

		if(listOnly) {	
			std::cout << "\t- " << deviceProperties.name << " (ID " << i << ")" << std::endl;
		}

		if(deviceProperties.totalGlobalMem > maxAvailableMemory)
		{
			maxAvailableMemory = deviceProperties.totalGlobalMem;
			deviceWithMostMemory = deviceProperties;
			chosenDeviceIndex = i;
		}
	}

	if(listOnly) {
		return deviceWithMostMemory;
	}

	if(forceGPU != -1) {
		chosenDeviceIndex = forceGPU;
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceWithMostMemory, chosenDeviceIndex));

	checkCudaErrors(cudaSetDevice(chosenDeviceIndex));
	std::cout << "Chose " << deviceWithMostMemory.name << " as main device." << std::endl;
#if PRINT_GPU_PROPERTIES

	std::cout << "This device supports CUDA Compute Capability v" << deviceWithMostMemory.major << "." << deviceWithMostMemory.minor << "." << std::endl;
	std::cout << std::endl;
	std::cout << "Other device info:" << std::endl;
	std::cout << "\t- Total global memory: " << deviceWithMostMemory.totalGlobalMem << std::endl;
	std::cout << "\t- Clock rate (KHz): " << deviceWithMostMemory.clockRate << std::endl;
	std::cout << "\t- Number of concurrent kernels: " << deviceWithMostMemory.concurrentKernels << std::endl;
	std::cout << "\t- Max grid size: (" << deviceWithMostMemory.maxGridSize[0] << ", " << deviceWithMostMemory.maxGridSize[1] << ", " << deviceWithMostMemory.maxGridSize[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block dimension: (" << deviceWithMostMemory.maxThreadsDim[0] << ", " << deviceWithMostMemory.maxThreadsDim[1] << ", " << deviceWithMostMemory.maxThreadsDim[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block: " << deviceWithMostMemory.maxThreadsPerBlock << std::endl;
	std::cout << "\t- Max threads per multiprocessor: " << deviceWithMostMemory.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "\t- Number of multiprocessors: " << deviceWithMostMemory.multiProcessorCount << std::endl;
	std::cout << "\t- Number of registers per block: " << deviceWithMostMemory.regsPerBlock << std::endl;
	std::cout << "\t- Number of registers per multiprocessor: " << deviceWithMostMemory.regsPerMultiprocessor << std::endl;
	std::cout << "\t- Total constant memory: " << deviceWithMostMemory.totalConstMem << std::endl;
	std::cout << "\t- Warp size measured in threads: " << deviceWithMostMemory.warpSize << std::endl;
	std::cout << "\t- Single to double precision performance ratio: " << deviceWithMostMemory.singleToDoublePrecisionPerfRatio << std::endl;
	std::cout << "\t- Shared memory per block: " << deviceWithMostMemory.sharedMemPerBlock << std::endl;
	std::cout << "\t- Shared memory per multiprocessor: " << deviceWithMostMemory.sharedMemPerMultiprocessor << std::endl;
	std::cout << "\t- L2 Cache size: " << deviceWithMostMemory.l2CacheSize << std::endl;
	std::cout << std::endl;
#endif

	return deviceWithMostMemory;
}
