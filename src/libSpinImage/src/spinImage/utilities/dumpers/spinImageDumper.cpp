#include "spinImageDumper.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <bitset>


#include <vector>
#include <lodepng.h>
#include <spinImage/cpu/types/QUICCIImages.h>

template<typename spinPixelType>
void performSpinDump(SpinImage::array<spinPixelType> descriptors, const std::experimental::filesystem::path &imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow) {
	size_t rowCount = (descriptors.length / imagesPerRow) + ((descriptors.length % imagesPerRow == 0) ? 0 : 1);
	std::cout << "Dumping " << rowCount << " rows containing " << descriptors.length << " images." << std::endl;

	unsigned int width = imagesPerRow * (spinImageWidthPixels + 1);
	size_t height = rowCount * (spinImageWidthPixels + 1);

	spinPixelType max = 0;
	size_t nonzeroPixelCount = 0;

	for (size_t image = 0; image < descriptors.length; image++)
	{
		for(size_t x = 0; x < spinImageWidthPixels; x++)
		{
			for(size_t y = 0; y < spinImageWidthPixels; y++)
			{
				size_t pixel_index = size_t(spinImageWidthPixels) * size_t(spinImageWidthPixels) * image + size_t(spinImageWidthPixels) * y + x;
				spinPixelType pixel_value = descriptors.content[pixel_index];
				if(pixel_value != 0) {
					nonzeroPixelCount++;
				}
				if(!std::isnan(pixel_value) && pixel_value != UINT32_MAX) {
					max = std::max(pixel_value, max);
				}
			}
		}
	}
	std::cout << "Image dumper: max is " << max << std::endl;
	std::cout << "Image dumper: nonzero pixel count is: " << nonzeroPixelCount << std::endl;

	if(max == 1) {
		std::cout << "WARNING: ignoring logarithmic image parameter, as maximum pixel value is 1 (would cause 0 division)." << std::endl;
	}

	std::vector<unsigned char> imageData;
	size_t pixelCount = width * height * 4;
	imageData.resize(pixelCount);
	size_t imageIndex = 0;

	// Initialise the images to default values.
	// Unused images will also be marked with a red cross
	for(size_t row = 0; row < rowCount; row++)
	{
		for(size_t col = 0; col < imagesPerRow; col++)
		{
			for (size_t x = 0; x < spinImageWidthPixels; x++)
			{
				for (size_t y = 0; y < spinImageWidthPixels; y++)
				{
					size_t pixelX = col * (spinImageWidthPixels + 1) + x;
					size_t pixelY = row * (spinImageWidthPixels + 1) + (spinImageWidthPixels - 1 - y);
					size_t pixelBaseIndex = 4 * (pixelX + width * pixelY);

					imageData[pixelBaseIndex + 0] = 255;
					imageData[pixelBaseIndex + 1] = 255;
					imageData[pixelBaseIndex + 2] = 255;
					imageData[pixelBaseIndex + 3] = 255;

					if (((x == y) || (x == spinImageWidthPixels - 1 - y)) && x < spinImageWidthPixels && y < spinImageWidthPixels)
					{
						imageData[pixelBaseIndex + 1] = 0;
						imageData[pixelBaseIndex + 2] = 0;
					}
				}
			}
			imageIndex++;
		}
	}

	imageIndex = 0;

	for(size_t row = 0; row < rowCount; row++)
	{
		for(size_t col = 0; col < imagesPerRow; col++)
		{
			if (imageIndex >= descriptors.length)
			{
				break; //stop once we have reached the final spin image
			}

			// Hacky way to mark image as empty. Used to display multiple sets of images.
			unsigned int topLeftPixel = descriptors.content[size_t(spinImageWidthPixels) * size_t(spinImageWidthPixels) * imageIndex];
			if(topLeftPixel != UINT32_MAX) {

				for (size_t x = 0; x < spinImageWidthPixels; x++) {
					for (size_t y = 0; y < spinImageWidthPixels; y++) {
						size_t pixel_index = size_t(spinImageWidthPixels) * size_t(spinImageWidthPixels) * imageIndex +
											 size_t(spinImageWidthPixels) * y + x;
						spinPixelType pixelValue = descriptors.content[pixel_index];
						float normalised;
						if (logarithmicImage && max != 1.0f) {
							normalised = (std::log(std::max(0.0f, float(pixelValue))) / std::log(float(max))) * 255.0f;
						} else {
							normalised = (std::max(0.0f, float(pixelValue)) / float(max)) * 255.0f;
						}
						unsigned char pixelByte = (unsigned char) unsigned(std::min(std::max(normalised, 0.0f), 255.0f));

						size_t pixelX = col * (spinImageWidthPixels + 1) + x;
						size_t pixelY = row * (spinImageWidthPixels + 1) +
										(spinImageWidthPixels - 1 - y); // Flip image because image coordinates

						size_t pixelBaseIndex = 4 * (pixelX + width * pixelY);

						if (!std::isnan(pixelValue)) {
							imageData[pixelBaseIndex + 0] = pixelByte;
							imageData[pixelBaseIndex + 1] = pixelByte;
							imageData[pixelBaseIndex + 2] = pixelByte;
							imageData[pixelBaseIndex + 3] = 255;
						} else {
							imageData[pixelBaseIndex + 0] = 255;
							imageData[pixelBaseIndex + 1] = 0;
							imageData[pixelBaseIndex + 2] = 0;
							imageData[pixelBaseIndex + 3] = 255;
						}
					}
				}
			}

			imageIndex++;
		}
	}

	std::cout << "Writing image file.. " << imageDestinationFile << std::endl;

	unsigned error = lodepng::encode(imageDestinationFile, imageData, width, height);

	if(error)
	{
		std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	}
}

void SpinImage::dump::descriptors(array<spinImagePixelType> hostDescriptors, std::experimental::filesystem::path imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow)
{
	performSpinDump<spinImagePixelType>(hostDescriptors, imageDestinationFile, logarithmicImage, imagesPerRow);
}

void SpinImage::dump::descriptors(array<radialIntersectionCountImagePixelType> hostDescriptors, std::experimental::filesystem::path imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow)
{
	performSpinDump<radialIntersectionCountImagePixelType> (hostDescriptors, imageDestinationFile, logarithmicImage, imagesPerRow);
}

void SpinImage::dump::descriptors(SpinImage::cpu::QUICCIImages hostDescriptors, std::experimental::filesystem::path imageDestinationFile, unsigned int imagesPerRow) {
	// Compute the number of images that should be inserted to separate the two series
	// If the number of rows fits the images exactly, an extra one is inserted for better clarity.
	size_t rowRemainder = hostDescriptors.imageCount % imagesPerRow;
	size_t fillerImageCount = (rowRemainder == 0) ? imagesPerRow : (imagesPerRow - rowRemainder);

	size_t totalImageCount = hostDescriptors.imageCount + fillerImageCount;

	const size_t pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;

	SpinImage::array<unsigned int> decompressedDesciptors;
	size_t imageTotalPixelCount = totalImageCount * pixelsPerImage;

	decompressedDesciptors.content = new unsigned int[imageTotalPixelCount];
	decompressedDesciptors.length = totalImageCount;

	const size_t uintsPerImage = pixelsPerImage / 32;

	size_t pixelIndex = 0;

	for(unsigned int imageIndex = 0; imageIndex < hostDescriptors.imageCount; imageIndex++) {
	    for(unsigned int chunkIndex = 0; chunkIndex < UINTS_PER_QUICCI; chunkIndex++) {
            unsigned int chunk = hostDescriptors.images[imageIndex][chunkIndex];
            std::bitset<32> entryBits(chunk);
            for(char bit = 0; bit < 32; bit++) {
                decompressedDesciptors.content[pixelIndex] = unsigned(int(entryBits[31 - bit]) * 255);
                pixelIndex++;
            }
        }
	}

	for(unsigned int emptyPixelIndex = 0; emptyPixelIndex < fillerImageCount * pixelsPerImage; emptyPixelIndex++) {
		decompressedDesciptors.content[pixelIndex] = UINT32_MAX;
		pixelIndex++;
	}

    performSpinDump<unsigned int>(decompressedDesciptors, imageDestinationFile, false, imagesPerRow);

    delete[] decompressedDesciptors.content;
}

void SpinImage::dump::descriptors(
        const std::vector<QuiccImage> &hostDescriptors,
        std::experimental::filesystem::path imageDestinationFile,
        unsigned int imagesPerRow) {
    size_t pixelIndex = 0;

    SpinImage::array<unsigned int> decompressedDescriptors = {
            hostDescriptors.size(),
            new unsigned int[hostDescriptors.size() * (sizeof(QuiccImage) / sizeof(unsigned int))]};

    for(unsigned int imageIndex = 0; imageIndex < hostDescriptors.size(); imageIndex++) {
        for(unsigned int chunkIndex = 0; chunkIndex < UINTS_PER_QUICCI; chunkIndex++) {
            unsigned int chunk = hostDescriptors.at(imageIndex)[chunkIndex];
            std::bitset<32> entryBits(chunk);
            for(char bit = 0; bit < 32; bit++) {
                decompressedDescriptors.content[pixelIndex] = unsigned(int(entryBits[31 - bit]) * 255);
                pixelIndex++;
            }
        }
    }

    performSpinDump<unsigned int>(decompressedDescriptors, imageDestinationFile, false, imagesPerRow);

    delete decompressedDescriptors.content;
}