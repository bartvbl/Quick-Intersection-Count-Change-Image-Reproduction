#include <arrrgh.hpp>
#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/IndexIO.h>
#include <spinImage/cpu/index/IndexQueryer.h>
#include <lodepng.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/dumpers/spinImageDumper.h>
#include <spinImage/utilities/fileutils.h>
#include <random>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <spinImage/cpu/index/SequentialIndexQueryer.h>
#include <json.hpp>
#include <tsl/ordered_map.h>
#include <fstream>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

int main(int argc, const char** argv) {
    arrrgh::parser parser("benchmarkindex", "Compare the time for looking up a randomly selected image from a directory of QUICCI dump files relative to iterating over the entire index.");
    const auto& indexDirectory = parser.add<std::string>(
            "index-directory", "The location of the directory containing the existing index.", '\0', arrrgh::Required, "");
    const auto& indexImageDirectory = parser.add<std::string>(
            "index-quicci-dump-directory", "The directory containing QUICCI dump files from which the index was generated.", '\0', arrrgh::Required, "");
    const auto& queryImageDirectory = parser.add<std::string>(
            "query-quicci-dump-directory", "The directory containing QUICCI dump files from which the query image should be drawn.", '\0', arrrgh::Required, "");
    const auto& outputDirectory = parser.add<std::string>(
            "output-directory", "A directory to where a JSON file containing measurements should be written to.", '\0', arrrgh::Optional, ".");
    const auto& overrideRandomSeed = parser.add<size_t>("random-seed", "Specify a random seed for the random number generator. The same seed used with the same index and dataset will yield the same results. Leaving this parameter unspecified will select a random seed.", '\0', arrrgh::Optional, 0);

    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

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



    SpinImage::index::debug::QueryRunInfo indexedRunInfo;
    SpinImage::index::debug::QueryRunInfo sequentialSingleThreadedRunInfo;
    SpinImage::index::debug::QueryRunInfo sequentialParallelRunInfo;

    std::random_device rd;
    size_t randomSeed = overrideRandomSeed.value() != 0 ? overrideRandomSeed.value() : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};

    std::cout << "Reading query image.." << std::endl;

    std::vector<std::experimental::filesystem::path> queryFiles = SpinImage::utilities::listDirectory(queryImageDirectory.value());
    std::sort(queryFiles.begin(), queryFiles.end());
    std::cout << "\tFound " << queryFiles.size() << " query image dump files." << std::endl;

    std::uniform_int_distribution<size_t> fileDistribution(0, queryFiles.size());
    size_t chosenFileIndex = fileDistribution(generator);
    std::experimental::filesystem::path chosenQueryFilePath = queryFiles.at(chosenFileIndex);
    std::cout << "\tChose " << chosenQueryFilePath << " as file to select an image from." << std::endl;

    SpinImage::cpu::QUICCIImages queryImages = SpinImage::read::QUICCImagesFromDumpFile(chosenQueryFilePath);
    std::uniform_int_distribution<size_t> queryImageDistribution(0, queryImages.imageCount);
    size_t chosenQueryImageIndex = queryImageDistribution(generator);
    QuiccImage chosenQueryImage = queryImages.images[chosenQueryImageIndex];
    BitCountMipmapStack(chosenQueryImage).print();
    std::cout << "\tChose image " << chosenQueryImageIndex << "/" << queryImages.imageCount << " from selected image file." << std::endl;

    std::cout << "Reading index metadata.." << std::endl;
    Index index = SpinImage::index::io::readIndex(indexDirectory.value());

    const unsigned int resultCount = 750;

    std::cout << "Querying index.." << std::endl;
    std::vector<SpinImage::index::QueryResult> searchResults = SpinImage::index::query(index, chosenQueryImage, resultCount, &indexedRunInfo);

    std::cout << "Querying dataset sequentially.." << std::endl;

    std::vector<std::experimental::filesystem::path> imageIndexDirectoryContents = SpinImage::utilities::listDirectory(indexImageDirectory.value());
    std::vector<SpinImage::index::QueryResult> sequentialSearchResults = SpinImage::index::sequentialQuery(indexImageDirectory.value(), chosenQueryImage, resultCount, 0, imageIndexDirectoryContents.size(), 1, &sequentialSingleThreadedRunInfo);
    std::cout << "Dumping results.." << std::endl;

    std::experimental::filesystem::path jsonPath = std::experimental::filesystem::path(outputDirectory.value()) / ("queryTimes_" + std::to_string(randomSeed) + ".json");
    json outJson;

    outJson["version"] = "v3";
    outJson["nodesPerBlock"] = NODES_PER_BLOCK;
    outJson["nodeSplitThreshold"] = NODE_SPLIT_THRESHOLD;
    outJson["randomSeed"] = randomSeed;
    outJson["fileIndex"] = chosenFileIndex;
    outJson["imageIndex"] = chosenQueryImageIndex;
    outJson["chosenFilePath"] = chosenQueryFilePath;
    outJson["chosenFileName"] = chosenQueryFilePath.filename();

    outJson["indexedQueryResults"] = {};
    outJson["indexedQueryResults"]["queryTime"] = indexedRunInfo.totalQueryTime;
    outJson["indexedQueryResults"]["threadCount"] = indexedRunInfo.threadCount;
    outJson["indexedQueryResults"]["distanceTimes"] = indexedRunInfo.distanceTimes;

    outJson["sequentialSerialResults"] = {};
    outJson["sequentialSerialResults"]["queryTime"] = sequentialSingleThreadedRunInfo.totalQueryTime;
    outJson["sequentialSerialResults"]["threadCount"] = sequentialSingleThreadedRunInfo.threadCount;
    outJson["sequentialSerialResults"]["distanceTimes"] = sequentialSingleThreadedRunInfo.distanceTimes;

    std::ofstream outFile(jsonPath);
    outFile << outJson.dump(4);
    outFile.close();

    std::cout << std::endl << "Done." << std::endl << std::endl;
    std::cout << "Query timings have been written to: " << jsonPath.string() << std::endl;
}