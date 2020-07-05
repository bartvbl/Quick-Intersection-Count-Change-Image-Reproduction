#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

const int imageCount = spinImageWidthPixels * spinImageWidthPixels + 1 - 2;
const int pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;

SpinImage::array<spinImagePixelType> generateEmptySpinImages(size_t imageCount);
SpinImage::array<radialIntersectionCountImagePixelType> generateEmptyRadialIntersectionCountImages(size_t imageCount);

SpinImage::array<spinImagePixelType> generateRepeatingTemplateSpinImage(
        spinImagePixelType patternPart0,
        spinImagePixelType patternPart1,
        spinImagePixelType patternPart2,
        spinImagePixelType patternPart3,
        spinImagePixelType patternPart4,
        spinImagePixelType patternPart5,
        spinImagePixelType patternPart6,
        spinImagePixelType patternPart7);
SpinImage::array<radialIntersectionCountImagePixelType> generateRepeatingTemplateRadialIntersectionCountImage(
        radialIntersectionCountImagePixelType patternPart0,
        radialIntersectionCountImagePixelType patternPart1,
        radialIntersectionCountImagePixelType patternPart2,
        radialIntersectionCountImagePixelType patternPart3,
        radialIntersectionCountImagePixelType patternPart4,
        radialIntersectionCountImagePixelType patternPart5,
        radialIntersectionCountImagePixelType patternPart6,
        radialIntersectionCountImagePixelType patternPart7);

SpinImage::array<spinImagePixelType> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage);
SpinImage::array<radialIntersectionCountImagePixelType> generateKnownRadialIntersectionCountImageSequence(const int imageCount, const int pixelsPerImage);