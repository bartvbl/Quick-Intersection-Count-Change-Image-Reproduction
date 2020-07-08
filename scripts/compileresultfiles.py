import json
import os
import os.path
import datetime
import xlwt
import pprint
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

# PREAMBLE, SCROLL DOWN FOR SETTINGS

# Some of the experimental runs contained multiple descriptor types.
# However, I found out later that some of these descriptor generation/testing implementations contained some bugs
# Which rendered the results of specific descriptors, but not the others, of specific result sets invalid.
# I therefore added these; they're a safeguard that results for specific descriptor types which is known to be faulty
# is not included in the produced results spreadsheet
# (in case they somehow slip the net)

# Last known fault in code: unsigned integer subtraction in RICI comparison function
# Date threshold corresponds to this commit
def isQsiResultValid(fileCreationDateTime, resultJson):
    riciResultsValidAfter = datetime.datetime(year=2019, month=10, day=7, hour=15, minute=14, second=0, microsecond=0)
    return fileCreationDateTime > riciResultsValidAfter


# Last known fault in code: override object count did not match
# Date threshold corresponds to this commit
def isSiResultValid(fileCreationDateTime, resultJson):
    siResultsValidAfter = datetime.datetime(year=2019, month=9, day=26, hour=17, minute=28, second=0, microsecond=0)
    hasCorrectOverrideObjectCount = 'overrideObjectCount' in resultJson and resultJson['overrideObjectCount'] == 10
    hasCorrectSampleSetSize = resultJson['sampleSetSize'] == 10
    hasValidCreationDateTime = siResultsValidAfter < fileCreationDateTime

    return hasCorrectOverrideObjectCount or hasCorrectSampleSetSize

# Last known fault in code: none
def is3dscResultValid(fileCreationTime, resultJson):
    return True






# --- SETTINGS ---

# Master input directories
# Contains all data to be included in the spreadsheet
# Format: [path to JSON output directory]: ([human readable name of result set], [cluster result set was executed on])
#   Clusters used:
#       IDUN: Large cluster, though machines contain many different graphics cards. Must ABSOLUTELY NOT be used for time sensitive results
#       HEID: DGX-2 machine, contains 16 equivalent V100 cards. Used for time sensitive tests.
inputDirectories = {
    # RICI, measurements for matching and computation times (only tested on 4 added clutter objects)
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_noearlyexit/output': ('QSI, No early exit, 5 objects', 'HEID'),
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_withearlyexit/output': ('QSI, Early exit, 5 objects', 'HEID'),
    # RICI, measurements for matching performance
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output': ('Failed jobs from IDUN run', 'HEID'),
    '../input/results_computed_by_authors/IDUNRUNS/output_lotsofobjects_v4': ('primary QSI IDUN run', 'IDUN'),
    '../input/results_computed_by_authors/HEIDRUNS/output_seeds_qsi_v4_5objects_missing/output': ('re-run of 5 object QSI results that were missing raw files', 'HEID'),

    # SI, execution times and machine performance
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_10_objects_only/output': ('180 support angle, 10 objects', 'HEID'),
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output': ('180 support angle, 5 objects', 'HEID'),
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_180deg_si_missing/output': ('180 support angle, 10 objects', 'HEID'),
        # NOTE: 5 object sequence DOES NOT contribute to time measurements (the bit further down with lots of split() and merge() calls verifies this)
    '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_15': ('180 support angle, 1 & 5 objects', 'IDUN'),
    '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_10': ('180 support angle, 10 objects', 'IDUN'),
    '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_1': ('180 support angle, 1 object', 'IDUN'),
    # SI, testing matching performance of 60 degree support angle
    '../input/results_computed_by_authors/IDUNRUNS/output_smallsupportangle_lotsofobjects': ('60 support angle, primary', 'IDUN'),
    '../input/results_computed_by_authors/IDUNRUNS/output_qsifix_smallsupportangle_rerun': ('60 support angle, secondary', 'IDUN'),
    '../input/results_computed_by_authors/IDUNRUNS/output_supportanglechart60_si_v4_1': ('60 support angle, 1 object', 'IDUN'),
    '../input/results_computed_by_authors/IDUNRUNS/output_supportanglechart60_si_v4_5': ('60 support angle, 5 objects', 'IDUN'),
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_60deg_si_missing/output/': ('60 support angle, 10 objects', 'HEID'),
    '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_si_v4_60deg_5objects_missing/output/': ('60 support angle, 5 objects', 'HEID'),

    # 3DSC, execution time and matching performance are both measured
    '../input/results_computed_by_authors/HEIDRUNS/run1_3dsc_main/output/': ('3DSC', 'HEID'),
}

# The location where the master spreadsheet should be written to
outfile = '../output/master_spreadsheet.xls'
numCountsToIncludeInRawHistogramSpreadsheet = 100
fileMapLocation = '../output/filemap.json'

with open('../res/seeds_used_to_create_charts.txt') as seedFile:
    seedsUsedToCreateCharts = [str(x).strip() for x in seedFile.readlines()]


# Map of methods contained in the output files
methods = {
    'RICI': {
        'isValid': isQsiResultValid,
        'nameInJSONFile': 'qsi',
        'namePrefixInJSONFile': 'QSI',
        'generationTimings': ['total', 'meshScale', 'redistribution', 'generation'],
        'searchTimings': ['total', 'search']
    },
    'SI': {
        'isValid': isSiResultValid,
        'nameInJSONFile': 'si',
        'namePrefixInJSONFile': 'SI',
        'generationTimings': ['total', 'initialisation', 'sampling', 'generation'],
        'searchTimings': ['total', 'averaging', 'search']
    },
    '3DSC': {
        'isValid': is3dscResultValid,
        'nameInJSONFile': '3dsc',
        'namePrefixInJSONFile': '3DSC',
        'generationTimings': ['total', 'initialisation', 'sampling', 'pointCounting', 'generation'],
        'searchTimings': ['total', 'search']
    },
}


# Settings for clutter heatmaps
# Width and height of heatmap in pixels
heatmapSize = 256

rawInputDirectories = {
    'QSI': ['../input/results_computed_by_authors/HEIDRUNS/output_seeds_qsi_v4_5objects_missing/output/raw', 
            '../input/results_computed_by_authors/IDUNRUNS/output_lotsofobjects_v4/raw'],
    'SI': ['../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output/raw', 
           '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_15/raw'],
    '3DSC': ['../input/results_computed_by_authors/HEIDRUNS/run1_3dsc_main/output/raw'],
}
rawInputObjectCount = 5

clutterFileDirectories = ['../input/clutter_estimated_by_authors/clutter/']



# Distill the result set down to the entries we have all data for
removeSeedsWithMissingEntries = True

# Cut down the result set to a specific number of entries
resultSetSizeLimit = 1500
enableResultSetSizeLimit = True


# --- start of code ---

print()
print(' === Processing of experiment output files into the master spreadsheet ===')
print('This spreadsheet contains the exact data used to construct the charts in the paper')
print('Having a machine with 32GB of RAM is probably a necessity for running this')
print()

matplotlib.use('Qt5Agg')

# -- global initialisation --

# Maps seeds to a list of (dataset, value) tuples
seedmap_top_result = {}
seedmap_top10_results = {}
for methodName in methods:
    seedmap_top_result[methodName] = {}
    seedmap_top10_results[methodName] = {}

# -- code --

def extractExperimentSettings(loadedJson):
    settings = {}
    settings['boxSize'] = loadedJson['boxSize']
    settings['sampleObjectCounts'] = loadedJson['sampleObjectCounts']
    settings['sampleSetSize'] = loadedJson['sampleSetSize']
    settings['searchResultCount'] = loadedJson['searchResultCount']
    settings['spinImageSupportAngle'] = loadedJson['spinImageSupportAngle']
    settings['spinImageWidth'] = loadedJson['spinImageWidth']
    settings['spinImageWidthPixels'] = loadedJson['spinImageWidthPixels']
    if 'overrideObjectCount' in loadedJson:
        settings['overrideObjectCount'] = loadedJson['overrideObjectCount']
    else:
        settings['overrideObjectCount'] = max(loadedJson['sampleObjectCounts'])
    if 'descriptors' in loadedJson:
        settings['descriptors'] = loadedJson['descriptors']
    else:
        descriptors = []
        for method in methods:
            if methods[method]['namePrefixInJSONFile'] + 'histograms' in loadedJson:
                descriptors.append(methods[method]['nameInJSONFile'])
        settings['descriptors'] = descriptors
    settings['version'] = loadedJson['version']
    return settings


def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    # Filter out the raw output file directories
    originalFiles = [x for x in originalFiles if x != 'raw' and x != 'rawless']

    results = {
        'path': path,
        'results': {},
        'settings': {},
        'fileOriginMap': {}
    }

    jsonCache = {}

    ignoredLists = {}
    allResultsInvalid = {}

    for method in methods:
        results['results'][methods[method]['namePrefixInJSONFile']] = {}
        ignoredLists[method] = []
        allResultsInvalid[method] = False

    # Reading file contents
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex + 1) + '/' + str(len(originalFiles)), file + '        ', end='\r', flush=True)
        with open(os.path.join(path, file), 'r') as openFile:
            # Read JSON file
            try:
                fileContents = json.loads(openFile.read())
                fileContents['resultSetOrigin'] = os.path.join(path, file)
            except Exception as e:
                print('FAILED TO READ FILE: ' + str(file))
                print(e)
                continue
            jsonCache[file] = fileContents

            # Check validity of code by using knowledge about previous code changes
            filename_creation_time_part = file.split('_')[0]
            creation_time = datetime.datetime.strptime(filename_creation_time_part, "%Y-%m-%d %H-%M-%S")

            for method in methods:
                if not methods[method]['isValid'](creation_time, fileContents):
                    ignoredLists[method].append(file)
                    # If only a single result is deemed invalid,
                    # THE ENTIRE SET is marked as invalid for this descriptor type
                    # Data integrity is of the utmost importance, so I'm not taking any chances here.
                    allResultsInvalid[method] = True

    # Processing file contents
    previousExperimentSettings = None
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex + 1) + '/' + str(len(originalFiles)), file + '        ', end='\r', flush=True)

        # Read JSON file
        fileContents = jsonCache[file]

        # Check if settings are the same as other files in the folder
        currentExperimentSettings = extractExperimentSettings(fileContents)
        if previousExperimentSettings is not None:
            if currentExperimentSettings != previousExperimentSettings:
                # Any discrepancy here is a fatal exception. It NEEDS attention regardless
                raise Exception("Experiment settings mismatch in the same batch! File: " + file)
        previousExperimentSettings = currentExperimentSettings
        results['settings'] = currentExperimentSettings
        results['fileOriginMap'][fileContents['seed']] = os.path.join(path, file)

        for method in methods:
            # Check for other incorrect settings. Ignore files if detected
            if 0 in fileContents['imageCounts']:
                if file not in ignoredLists[method]:
                    ignoredLists[method].append(file)
            if fileContents['spinImageWidthPixels'] == 32:
                if file not in ignoredLists[method]:
                    ignoredLists[method].append(file)

            # Beauty checks
            if file not in ignoredLists[method] and allResultsInvalid[method]:
                ignoredLists[method].append(file)

            containsResultsForMethod = ('descriptors' in fileContents and methods[method]['nameInJSONFile']
                                        in fileContents['descriptors']) or methods[method]['namePrefixInJSONFile'] + 'histograms' in fileContents

            # Sanity checks are done. We can now add any remaining valid entries to the result lists
            if not file in ignoredLists[method] and not allResultsInvalid[method] and containsResultsForMethod:
                results['results'][methods[method]['namePrefixInJSONFile']][str(fileContents['seed'])] = fileContents

    print()

    results['settings'] = previousExperimentSettings

    return results


def objects(count):
    if count > 1:
        return 'Objects'
    else:
        return 'Object'



print('Loading raw data files..')
loadedRawResults = {}
for method in methods:
    loadedRawResults[methods[method]['namePrefixInJSONFile']] = {}

for algorithm in rawInputDirectories:
    for path in rawInputDirectories[algorithm]:
        print('Loading raw directory:', path)
        rawFiles = os.listdir(path)
        for fileIndex, file in enumerate(rawFiles):
            print(str(fileIndex + 1) + '/' + str(len(rawFiles)), file + '        ', end='\r', flush=True)
            seed = file.split('.')[0].split("_")[2]
            if not seed in loadedRawResults[algorithm]:
                with open(os.path.join(path, file), 'r') as openFile:
                    # Read JSON file
                    try:
                        rawFileContents = json.loads(openFile.read())
                        loadedRawResults[algorithm][seed] = rawFileContents[algorithm][str(rawInputObjectCount)]
                    except Exception as e:
                        print('FAILED TO READ FILE: ' + str(file))
                        print(e)
                        continue
        print()


print()
print('Loading input data files..')
loadedResults = {}
for directory in inputDirectories.keys():
    print('Loading directory:', directory)
    loadedResults[directory] = loadOutputFileDirectory(directory)


def filterResultSet(resultSet, index):
    out = copy.deepcopy(resultSet)

    for method in methods:
        sampleGenerationString = methods[method]['namePrefixInJSONFile'] + 'SampleGeneration'
        searchString = methods[method]['namePrefixInJSONFile'] + 'Search'
        histogramsString = methods[method]['namePrefixInJSONFile'] + 'histograms'

        if sampleGenerationString in out['runtimes']:
            for timingMeasurement in methods[method]['generationTimings']:
                out['runtimes'][sampleGenerationString][timingMeasurement] \
                    = [out['runtimes'][sampleGenerationString][timingMeasurement][index]]

        if searchString in out['runtimes']:
            for timingMeasurement in methods[method]['searchTimings']:
                out['runtimes'][searchString][timingMeasurement] = [out['runtimes'][searchString][timingMeasurement][index]]

        if histogramsString in out:
            # Older result dump files use an integer index as a way to specify which object count the results belong to
            # Newer dump files use the latter format, where it specifically lists the object count used
            # This if statement automatically switches between these variants
            if str(index) in out[histogramsString]:
                out[histogramsString] = {'0': out[histogramsString][str(index)]}
            else:
                out[histogramsString] = {'0': out[histogramsString][str(out['sampleObjectCounts'][index]) + ' objects']}

    return out


def split(directory):
    print('Splitting', directory)
    global loadedResults

    result = loadedResults[directory]
    del loadedResults[directory]

    setMeta = inputDirectories[directory]
    del inputDirectories[directory]

    for itemCountIndex, itemCount in enumerate(result['settings']['sampleObjectCounts']):
        out = {'results': {}, 'settings': {}, 'fileOriginMap': {}}
        for method in methods:
            out['results'][methods[method]['namePrefixInJSONFile']] = {}

        out['settings'] = result['settings'].copy()
        out['settings']['sampleObjectCounts'] = [itemCount]
        out['fileOriginMap'] = result['fileOriginMap'].copy()

        for method in methods:
            for seed in result['results'][methods[method]['namePrefixInJSONFile']]:
                out['results'][methods[method]['namePrefixInJSONFile']][seed] = \
                    filterResultSet(result['results'][methods[method]['namePrefixInJSONFile']][seed], itemCountIndex)

        newDirectoryName = directory + ' (' + str(itemCount) + ' objects)'

        loadedResults[newDirectoryName] = out
        inputDirectories[newDirectoryName] = (setMeta[0] + ' (' + str(itemCount) + ' objects)', setMeta[1])


def merge(directory1, directory2, newdirectoryName, newDirectoryClusterName):
    global loadedResults
    global inputDirectories

    print('Merging', directory2, 'into', directory1)

    directory1_contents = loadedResults[directory1]
    directory2_contents = loadedResults[directory2]

    del loadedResults[directory1]
    del loadedResults[directory2]
    del inputDirectories[directory1]
    del inputDirectories[directory2]

    if directory1_contents['settings'] != directory2_contents['settings']:
        print(
            'WARNING: Directories %s and %s have different generation settings, and may therefore not be compatible to be merged!' % (
            directory1, directory2))
        print('Directory 1:', directory1_contents['settings'])
        print('Directory 2:', directory2_contents['settings'])

    combinedResults = {'results': {}, 'settings': directory1_contents['settings'], 'fileOriginMap': directory1_contents['fileOriginMap']}

    for method in methods:
        combinedResults['results'][methods[method]['namePrefixInJSONFile']] = {}

    # Initialising with the original results
    for method in methods:
        combinedResults['results'][methods[method]['namePrefixInJSONFile']] = \
            directory1_contents['results'][methods[method]['namePrefixInJSONFile']]

    additionCount = 0

    # Now we merge any missing results into it
    for type in directory2_contents['results']:
        for seed in directory2_contents['results'][type].keys():
            if seed not in combinedResults['results'][type]:
                combinedResults['results'][type][seed] = directory2_contents['results'][type][seed]
                additionCount += 1

    for seed in directory2_contents['fileOriginMap'].keys():
        if seed not in combinedResults['fileOriginMap']:
            combinedResults['fileOriginMap'][seed] = directory2_contents['fileOriginMap'][seed]

    loadedResults[newdirectoryName] = combinedResults
    inputDirectories[newdirectoryName] = (newdirectoryName, newDirectoryClusterName)

    print('\tAdded', additionCount, 'new values')
    return additionCount

# Small hack, but silences a warning that does not apply here
loadedResults['../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output']['settings']['overrideObjectCount'] = 10

print('\nRestructuring datasets..\n')
split('../input/results_computed_by_authors/IDUNRUNS/output_smallsupportangle_lotsofobjects')
split('../input/results_computed_by_authors/IDUNRUNS/output_qsifix_smallsupportangle_rerun')
split('../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_15')
split('../input/results_computed_by_authors/IDUNRUNS/output_lotsofobjects_v4')
split('../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output')
print()

# QSI 1 object
merge('../input/results_computed_by_authors/IDUNRUNS/output_lotsofobjects_v4 (1 objects)', '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output (1 objects)',
      'QSI, 1 object', 'HEID + IDUN')

# QSI 5 objects
merge('../input/results_computed_by_authors/HEIDRUNS/output_seeds_qsi_v4_5objects_missing/output', '../input/results_computed_by_authors/IDUNRUNS/output_lotsofobjects_v4 (5 objects)', 'QSI primary intermediate', 'HEID + IDUN')
merge('QSI primary intermediate', '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output (5 objects)',
      'QSI, 5 objects', 'HEID + IDUN')

# QSI 10 objects
merge('../input/results_computed_by_authors/IDUNRUNS/output_lotsofobjects_v4 (10 objects)', '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output (10 objects)',
      'QSI, 10 objects', 'HEID + IDUN')

# SI 180 degrees, 1 object
merge('../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_1', '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_15 (1 objects)',
      'SI 180 degrees, 1 object', 'IDUN')

# SI 180 degrees, 5 objects
additionCount = merge('../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output',
                      '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_15 (5 objects)', 'SI 180 degrees, 5 objects', 'HEID')
# this merge is mainly to remove the dataset from the input batch. We ultimately want the HEIDRUNS results exclusively because
# we use these results to compare runtimes
assert (additionCount == 0)

# SI 180 degrees, 10 objects
merge('../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_lotsofobjects_10_objects_only/output', '../input/results_computed_by_authors/IDUNRUNS/output_mainchart_si_v4_10',
      'SI 180 degrees, 10 objects', 'HEID + IDUN')
merge('SI 180 degrees, 10 objects', '../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_180deg_si_missing/output',
      'SI 180 degrees, 10 objects', 'HEID + IDUN')

# SI 60 degrees, 1 object
merge('../input/results_computed_by_authors/IDUNRUNS/output_supportanglechart60_si_v4_1',
      '../input/results_computed_by_authors/IDUNRUNS/output_smallsupportangle_lotsofobjects (1 objects)', 'SI 60 degrees, 1 object intermediate', 'IDUN')
merge('SI 60 degrees, 1 object intermediate', '../input/results_computed_by_authors/IDUNRUNS/output_qsifix_smallsupportangle_rerun (1 objects)',
      'SI 60 degrees, 1 object', 'IDUN')

# SI 60 degrees, 5 objects
merge('../input/results_computed_by_authors/HEIDRUNS/output_qsifix_si_v4_60deg_5objects_missing/output/', '../input/results_computed_by_authors/IDUNRUNS/output_supportanglechart60_si_v4_5', 'SI 60 degrees, 5 objects, intermediate', 'HEID + IDUN')
merge('SI 60 degrees, 5 objects, intermediate',
      '../input/results_computed_by_authors/IDUNRUNS/output_smallsupportangle_lotsofobjects (5 objects)', 'SI 60 degrees, 5 objects', 'HEID + IDUN')

# SO 60 degrees, 10 objects
merge('../input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_60deg_si_missing/output/',
      '../input/results_computed_by_authors/IDUNRUNS/output_smallsupportangle_lotsofobjects (10 objects)', 'SI 60 deg 10 objects intermediate',
      'HEID + IDUN')
merge('SI 60 deg 10 objects intermediate', '../input/results_computed_by_authors/IDUNRUNS/output_qsifix_smallsupportangle_rerun (10 objects)',
      'SI 60 degrees, 10 objects', 'HEID + IDUN')

print()

print('\nProcessing..\n')

seedSet = set()
for directory in inputDirectories.keys():
    for method in methods:
        for seed in loadedResults[directory]['results'][methods[method]['namePrefixInJSONFile']].keys():
            seedSet.add(seed)
seedList = [x for x in seedSet]

print('Found', len(seedSet), 'seeds in result sets')



if removeSeedsWithMissingEntries:
    print('\nRemoving missing entries..')
    missingSeeds = []

    for directory in loadedResults:
        cutCount = 0
        for seed in seedList:
            for method in methods:
                if len(loadedResults[directory]['results'][methods[method]['namePrefixInJSONFile']]) > 0:
                    if not seed in loadedResults[directory]['results'][methods[method]['namePrefixInJSONFile']]:
                        missingSeeds.append(seed)
                        cutCount += 1
        print(directory, '- removed seed count:', cutCount)

    print('Detected', len(set(missingSeeds)), 'unique seeds with missing entries. Removing..')

    for missingSeed in missingSeeds:
        for directory in loadedResults:
            for method in methods:
                if missingSeed in loadedResults[directory]['results'][methods[method]['namePrefixInJSONFile']]:
                    del loadedResults[directory]['results'][methods[method]['namePrefixInJSONFile']][missingSeed]
        if missingSeed in seedList:
            del seedList[seedList.index(missingSeed)]

print('\nLoading clutter files..')
clutterFileMap = {}
for clutterFileDirectory in clutterFileDirectories:
    print('Reading directory', clutterFileDirectory)
    clutterFiles = os.listdir(clutterFileDirectory)
    for clutterFileIndex, clutterFile in enumerate(clutterFiles):
        print(str(clutterFileIndex + 1) + '/' + str(len(clutterFiles)), clutterFile + '        ', end='\r', flush=True)
        with open(os.path.join(clutterFileDirectory, clutterFile), 'r') as openFile:
            # Read JSON file
            try:
                clutterFileContents = json.loads(openFile.read())
                seed = clutterFileContents['sourceFile'].split('.')[0].split('_')[2]
                clutterFileMap[seed] = clutterFileContents
            except Exception as e:
                print('FAILED TO READ FILE: ' + str(file))
                print(e)
                continue
    print()

# Find the seeds for which all input sets have data
print('\nComputing condensed seed set')
rawSeedList = loadedRawResults[methods[[x for x in methods.keys()][0]]['namePrefixInJSONFile']].keys()
print('Starting raw RICI seed set size:', len(rawSeedList))
for method in methods:
    rawSeedList = [x for x in rawSeedList if x in loadedRawResults[methods[method]['namePrefixInJSONFile']].keys()]
    print('Merged with ' + method + ':', len(rawSeedList))
rawSeedList = [x for x in rawSeedList if x in seedList]
print('Merged with seedList:', len(rawSeedList))
rawSeedList = [x for x in rawSeedList if x in clutterFileMap.keys()]
print('Merged with clutter file map:', len(rawSeedList))
rawSeedList = [x for x in rawSeedList if x in seedsUsedToCreateCharts]
print('Merged with seed list used for paper:', len(rawSeedList))
print('Seeds remaining:', len(rawSeedList))



# We want to make sure our dataset contains entries for ALL data points
seedList = rawSeedList

if enableResultSetSizeLimit:
    seedList = [x for index, x in enumerate(seedList) if index < resultSetSizeLimit]

print()
print('Reduced result set to size', len(seedList))
print()




print()
print('Dumping file map..')

correspondingFileMap = {}

directoriesToDump = [
    '../input/results_computed_by_authors/HEIDRUNS/run1_3dsc_main/output/',
    'QSI, 1 object',
    'QSI, 5 objects',
    'QSI, 10 objects',
    'SI 180 degrees, 1 object',
    'SI 180 degrees, 5 objects',
    'SI 180 degrees, 10 objects']

for directory in directoriesToDump:
    resultSet = loadedResults[directory]
    for method in methods:
        if not method in correspondingFileMap:
            correspondingFileMap[method] = {}
        methodName = methods[method]['namePrefixInJSONFile']
        for seed in resultSet['results'][methodName].keys():
            entry = resultSet['results'][methodName][seed]
            sourceFile = ''
            if int(seed) in resultSet['fileOriginMap']:
                sourceFile = resultSet['fileOriginMap'][int(seed)]
            for objectCount in entry['sampleObjectCounts']:
                if not str(objectCount) in correspondingFileMap[method]:
                    correspondingFileMap[method][str(objectCount)] = {}
                correspondingFileMap[method][str(objectCount)][str(seed)] = sourceFile

with open(fileMapLocation, 'w') as fileMapFile:
    fileMapFile.write(json.dumps(correspondingFileMap, indent=4))
print('Success.')
print()



# Create heatmap histograms
histograms = {}
for method in methods:
    histograms[method] = np.zeros(shape=(heatmapSize, heatmapSize), dtype=np.int64)

print('Computing histograms..')
histogramEntryCount = 0
for rawSeedIndex, rawSeed in enumerate(rawSeedList):
    print(str(rawSeedIndex + 1) + '/' + str(len(rawSeedList)) + " processed", end='\r', flush=True)

    clutterValues = clutterFileMap[rawSeed]['clutterValues']

    histogramEntryCount += len(clutterValues)

    for method in methods:
        ranks = loadedRawResults[methods[method]['namePrefixInJSONFile']][rawSeed]

        if not(len(clutterValues) == len(ranks)):
            print('WARNING: batch size mismatch at seed', rawSeed , '!', 'method ' + method, [len(clutterValues), len(ranks)])

        for i in range(0, len(clutterValues)):
            clutterValue = clutterValues[i]
            index = ranks[i]

            # Apparently some NaN in there
            if clutterValue is None:
                continue

            xBin = int((1.0 - clutterValue) * heatmapSize)
            if xBin >= heatmapSize:
                continue

            yBin = index
            if yBin < heatmapSize:
                histograms[method][heatmapSize - 1 - yBin, xBin] += 1

print('Histogram computed over', histogramEntryCount, 'values')

for method in methods:
    histograms[method] = np.log10(np.maximum(histograms[method], 0.1))



extent = [0, heatmapSize, 0, heatmapSize]

# Plot heatmap
plt.clf()

colorbar_ticks = np.arange(0, 8, 1)
total_minimum_value = min([np.amin(histograms[x]) for x in histograms])
total_maximum_value = max([np.amax(histograms[x]) for x in histograms])
print('range:', total_minimum_value, total_maximum_value)
#total_minimum_value = -1.0
#total_maximum_value = 7.168897592566977
normalisation = colors.Normalize(vmin=total_minimum_value,vmax=total_maximum_value)

horizontal_ticks_real_coords = np.arange(0,256,25.599*2.0)
horizontal_ticks_labels = [("%.1f" % x) for x in np.arange(0,1.1,0.2)]

for figureIndex, method in enumerate(methods):
    plot = plt.figure(figureIndex + 1)
    plt.title(method.upper())
    plt.ylabel('rank')
    plt.xlabel('clutter percentage')
    image = plt.imshow(histograms[method], extent=extent, cmap='nipy_spectral', norm=normalisation)
    plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)

    # Final chart gets the legend
    if figureIndex + 1 == len(methods.keys()):
        cbar = plt.colorbar(image, ticks=colorbar_ticks)
        cbar.ax.set_yticklabels(["{:.0E}".format(x) for x in np.power(10, colorbar_ticks)])
        cbar.set_label('Sample count', rotation=90)

    plot.show()

print()
print('Heatmap generation complete, press enter to dump spreadsheets.')
print('(which will also close the heatmap windows)')
input()

# -- Dump to spreadsheet --
#methods[method]['namePrefixInJSONFile']
print('Dumping spreadsheet..')

book = xlwt.Workbook(encoding="utf-8")

# Create data page for dataset settings table
experimentSheet = book.add_sheet("Experiment Overview")
allColumns = set()
for directoryIndex, directory in enumerate(inputDirectories.keys()):
    result = loadedResults[directory]
    allColumns = allColumns.union(set(result['settings']))

# Overview table headers
for keyIndex, key in enumerate(allColumns):
    experimentSheet.write(0, keyIndex + 1, str(key))
experimentSheet.write(0, len(allColumns) + 1, 'Cluster')
for index, method in enumerate(methods):
    experimentSheet.write(0, len(allColumns) + index + 2, method + ' Count')

# Overview table contents
for directoryIndex, directory in enumerate(inputDirectories.keys()):
    directoryName, cluster = inputDirectories[directory]
    experimentSheet.write(directoryIndex + 1, 0, directoryName)
    result = loadedResults[directory]
    for keyIndex, key in enumerate(allColumns):
        if key in result['settings']:
            experimentSheet.write(directoryIndex + 1, keyIndex + 1, str(result['settings'][key]))
        else:
            experimentSheet.write(directoryIndex + 1, keyIndex + 1, ' ')

    experimentSheet.write(directoryIndex + 1, len(allColumns) + 1, cluster)
    for columnIndex, method in enumerate(methods):
        experimentSheet.write(directoryIndex + 1, len(allColumns) + columnIndex + 2,
              len([x for x in result['results'][methods[method]['namePrefixInJSONFile']] if x in seedList]))

# Sheets
sheets = {}
for method in methods:
    sheets[method] = {}
for method in methods:
    sheets[method]['top0'] = book.add_sheet("Rank 0 " + method + " results")
for method in methods:
    sheets[method]['top10'] = book.add_sheet("Top 10 " + method + " results")
for method in methods:
    sheets[method]['generationSpeed'] = book.add_sheet(method + " Generation Times")
for method in methods:
    sheets[method]['comparisonSpeed'] = book.add_sheet(method + " Comparison Times")

vertexCountSheet = book.add_sheet("Reference Image Count")
totalVertexCountSheet = book.add_sheet("Total Image Count")
totalTriangleCountSheet = book.add_sheet("Total Triangle Count")


# Write initial columns

for method in methods:
    sheets[method]['top0'].write(0, 0, 'seed')
    sheets[method]['top10'].write(0, 0, 'seed')
    sheets[method]['generationSpeed'].write(0, 0, 'seed')
    sheets[method]['comparisonSpeed'].write(0, 0, 'seed')

vertexCountSheet.write(0, 0, 'seed')
totalVertexCountSheet.write(0, 0, 'seed')
totalTriangleCountSheet.write(0, 0, 'seed')

for method in methods:
    for seedIndex, seed in enumerate(seedList):
        sheets[method]['top0'].write(seedIndex + 1, 0, seed)
        sheets[method]['top10'].write(seedIndex + 1, 0, seed)
        sheets[method]['generationSpeed'].write(seedIndex + 1, 0, seed)
        sheets[method]['comparisonSpeed'].write(seedIndex + 1, 0, seed)

for seedIndex, seed in enumerate(seedList):
    vertexCountSheet.write(seedIndex + 1, 0, seed)
    totalVertexCountSheet.write(seedIndex + 1, 0, seed)
    totalTriangleCountSheet.write(seedIndex + 1, 0, seed)


# seed column is 0, data starts at column 1
currentColumn = 1

for directoryIndex, directory in enumerate(inputDirectories.keys()):
    for methodIndex, method in enumerate(methods):
        resultSet = loadedResults[directory]
        directoryName, _ = inputDirectories[directory]

        sampleGenerationString = methods[method]['namePrefixInJSONFile'] + 'SampleGeneration'
        searchString = methods[method]['namePrefixInJSONFile'] + 'Search'
        histogramsString = methods[method]['namePrefixInJSONFile'] + 'histograms'

        # Writing column headers
        for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
            columnHeader = directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
                len(resultSet['settings']['sampleObjectCounts'])) + ')'

            sheets[method]['top0'].write(0, currentColumn + sampleCountIndex, columnHeader)
            sheets[method]['top10'].write(0, currentColumn + sampleCountIndex, columnHeader)
            sheets[method]['generationSpeed'].write(0, currentColumn + sampleCountIndex, columnHeader)
            sheets[method]['comparisonSpeed'].write(0, currentColumn + sampleCountIndex, columnHeader)

            if methodIndex == 0:
                vertexCountSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
                totalVertexCountSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
                totalTriangleCountSheet.write(0, currentColumn + sampleCountIndex, columnHeader)

        for seedIndex, seed in enumerate(seedList):
            if seed in resultSet['results'][methods[method]['namePrefixInJSONFile']]:
                for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):

                    # Top 1 performance
                    entry = resultSet['results'][methods[method]['namePrefixInJSONFile']][seed]
                    totalImageCount = entry['imageCounts'][0]
                    experimentIterationCount = len(resultSet['settings']['sampleObjectCounts'])
                    percentageAtPlace0 = 0
                    totalImageCountInTop10 = 0
                    indexNameString = str(resultSet['settings']['sampleObjectCounts'][sampleCountIndex]) + ' objects'
                    if str(sampleCountIndex) in entry[histogramsString]:
                        if '0' in entry[histogramsString][str(sampleCountIndex)]:
                            percentageAtPlace0 = float(entry[histogramsString][str(sampleCountIndex)]['0']) / float(totalImageCount)
                        totalImageCountInTop10 = sum(
                            [entry[histogramsString][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                             str(x) in entry[histogramsString][str(sampleCountIndex)]])
                    else:
                        if '0' in entry[histogramsString][indexNameString]:
                            percentageAtPlace0 = float(entry[histogramsString][indexNameString]['0']) / float(
                                totalImageCount)
                        totalImageCountInTop10 = sum(
                            [entry[histogramsString][indexNameString][str(x)] for x in range(0, 10) if
                             str(x) in entry[histogramsString][indexNameString]])
                    sheets[method]['top0'].write(seedIndex + 1, currentColumn + sampleCountIndex, percentageAtPlace0)

                    # Top 10 performance

                    percentInTop10 = float(totalImageCountInTop10) / float(totalImageCount)
                    sheets[method]['top10'].write(seedIndex + 1, currentColumn + sampleCountIndex, percentInTop10)

                    # generation execution time
                    generationTime = entry['runtimes'][sampleGenerationString]['total'][sampleCountIndex]
                    sheets[method]['generationSpeed'].write(seedIndex + 1, currentColumn + sampleCountIndex, generationTime)

                    # search execution time
                    comparisonTime = entry['runtimes'][searchString]['total'][sampleCountIndex]
                    sheets[method]['comparisonSpeed'].write(seedIndex + 1, currentColumn + sampleCountIndex, comparisonTime)

                    # Vertex count sanity check
                    vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
                    totalVertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                                sum(entry['imageCounts'][0:sampleObjectCount]))
                    totalTriangleCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                                  sum(entry['vertexCounts'][0:sampleObjectCount]) / 3)
            else:
                for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                    sheets[method]['top0'].write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                    sheets[method]['top10'].write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                    sheets[method]['generationSpeed'].write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                    sheets[method]['comparisonSpeed'].write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')

    # Moving on to the next column
    currentColumn += len(resultSet['settings']['sampleObjectCounts'])

# beauty addition.. Cuts off final column
for seedIndex, seed in enumerate(seedList + ['dummy entry for final row']):
    for method in methods:
        sheets[method]['top0'].write(seedIndex, currentColumn, ' ')
        sheets[method]['top10'].write(seedIndex, currentColumn, ' ')
        sheets[method]['generationSpeed'].write(seedIndex, currentColumn, ' ')
        sheets[method]['comparisonSpeed'].write(seedIndex, currentColumn, ' ')

    vertexCountSheet.write(seedIndex, currentColumn, ' ')
    totalVertexCountSheet.write(seedIndex, currentColumn, ' ')
    totalTriangleCountSheet.write(seedIndex, currentColumn, ' ')

book.save(outfile)

print('Complete.')
