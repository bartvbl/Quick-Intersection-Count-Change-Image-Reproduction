import json
import os
import subprocess
import random

from simple_term_menu import TerminalMenu

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)
    subprocess.run(command, shell=True, check=False, cwd=working_directory)

def ask_for_confirmation(message):
    confirmation_menu = TerminalMenu(["yes", "no"], title=message)
    choice = confirmation_menu.show()
    return choice == 0

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download SHREC 2017 3D shape dataset (7.3GB download, ~50GB extracted)",
        "Download experiment results generated by authors (~1.5GB download, ~12GB extracted)",
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show()
        os.makedirs('input/download/', exist_ok=True)

        if choice == 0:
            if not os.path.isfile('input/download/SHREC17.7z') or ask_for_confirmation('It appears the SHREC 2017 dataset has already been downloaded. Would you like to download it again?'):
                print('Downloading SHREC 2017 dataset..')
                run_command_line_command('wget --output-document SHREC17.7z https://data.mendeley.com/datasets/ysh8p862v2/1/files/607f79cd-74c9-4bfc-9bf1-6d75527ae516/SHREC17.7z?dl=1', 'input/download/')
            print()
            os.makedirs('input/SHREC17', exist_ok=True)
            run_command_line_command('p7zip -k -d download/SHREC17.7z', 'input/')
            print('Download and extraction complete. You may now delete the file input/download/SHREC17.7z if you need the disk space.')
            print()

        if choice == 1:
            print('Downloading author generated results')
            print()
            run_command_line_command('p7zip -k -d download/clutter_estimated_by_authors.7z', 'input/')

            run_command_line_command('p7zip -k -d download/results_computed_by_authors.7z', 'input/')
        if choice == 2:
            return

def installDependenciesMenu():
    install_menu = TerminalMenu([
        "Install all dependencies except CUDA",
        "Install CUDA (through APT)",
        "back"], title='---------------- Install Dependencies ----------------')

    while True:
        choice = install_menu.show()

        if choice == 0:
            run_command_line_command('sudo apt install cmake python3 python3-pip libpcl-dev g++-7 gcc-7 wget p7zip')
            run_command_line_command('sudo pip3 install console-menu xlwt numpy matplotlib pillow')
            print()
        if choice == 1:
            run_command_line_command('sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev')
            print()
        if choice == 2:
            return

def compileProject():
    print('This project uses cmake for generating its makefiles.')
    print('It has a tendency to at times be unable to find an installed CUDA compiler.')
    print('Also, depending on which version of CUDA you have installed, you may need')
    print('to change the version of GCC/G++ used for compatibility reasons.')
    print('If either of these occurs, modify the paths at the top of the following file: ')
    print('    src/clutterbox/CMakeLists.txt')
    print()

    os.makedirs('src/clutterbox/build', exist_ok=True)

    compileProjectMenu = TerminalMenu([
        "Run cmake (must run before make)",
        "Run make",
        "back"], title='------------------- Compile Project -------------------')

    while True:
        choice = compileProjectMenu.show()

        if choice == 0:
            run_command_line_command('rm ./*', 'src/clutterbox/build')
            run_command_line_command('cmake ..', 'src/clutterbox/build')
        if choice == 1:
            run_command_line_command('make -j 4', 'src/clutterbox/build')
        if choice == 2:
            return

def runSpreadsheetBuilder():
    if ask_for_confirmation('The spreadsheet construction script will consume around 16GB of RAM.\nYou should close any applications to ensure you do not run out of memory.\nContinue?'):
        print()
        run_command_line_command('python3 buildspreadsheets.py', 'scripts/')


activeDescriptors = ['rici', 'si', '3dsc']
activeObjectCounts = ['1', '5', '10']
spinImageSupportAngle = 180
gpuID = 0
seedFileLocation = 'res/seeds_used_for_clutterbox_experiments.txt'
with open(seedFileLocation) as seedFile:
    random_seeds = [line.strip() for line in seedFile]

clutterSourceDumpFileDirectory = 'input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_withearlyexit/output'
estimatedClutterDirectory = 'input/clutter_estimated_by_authors/clutter/'
clutterFileMap = None



def executeClutterboxExperiment(randomSeed, matchVisualisationDirectory = None, matchVisualisationThreshold = 0, sceneOBJDumpDirectory = None):
    visualisationParameters = ''
    if matchVisualisationDirectory is not None:
        visualisationParameters = '--dump-matches-visualisation-obj-directory=' + matchVisualisationDirectory + ' ' \
                                  '--dump-matches-visualisation-obj-descriptors=' + ','.join(activeDescriptors) + ' ' \
                                  '--dump-matches-visualisation-obj-threshold=' + str(matchVisualisationThreshold) + ' '
    sceneOBJParameter = ''
    if sceneOBJDumpDirectory is not None:
        sceneOBJParameter = '--scene-obj-file-dump-directory=' + sceneOBJDumpDirectory + ' '
    run_command_line_command('src/clutterbox/build/clutterbox '
                             '--box-size=1 '
                             '--output-directory=output/clutterbox_results/ '
                             '--source-directory=input/SHREC17/ '
                             '--object-counts=' + ','.join(activeObjectCounts) + ' '
                             '--override-total-object-count=10 '
                             '--descriptors=' + ','.join(activeDescriptors) + ' '
                             '--support-radius=0.3 '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--force-seed=' + str(randomSeed) + ' '
                             '--spin-image-support-angle-degrees=' + str(spinImageSupportAngle) + ' '
                             '--3dsc-min-support-radius=0.048 '
                             '--3dsc-point-density-radius=0.096 '
                             '--dump-raw-search-results ' +
                             visualisationParameters +
                             sceneOBJParameter)

def configureActiveDescriptors():
    while True:
        run_menu = TerminalMenu([
            "Generate results for Radial Intersection Count Image: " + ("enabled" if "rici" in activeDescriptors else "disabled"),
            "Generate results for Spin Image: " + ("enabled" if "si" in activeDescriptors else "disabled"),
            "Generate results for 3D Shape Context: " + ("enabled" if "3dsc" in activeDescriptors else "disabled"),
            "done"], title='-- Configure descriptors to be tested --')
        choice = run_menu.show()
        if choice == 0:
            if "rici" in activeDescriptors:
                activeDescriptors.remove("rici")
            else:
                activeDescriptors.append("rici")
        if choice == 1:
            if "si" in activeDescriptors:
                activeDescriptors.remove("si")
            else:
                activeDescriptors.append("si")
        if choice == 2:
            if "3dsc" in activeDescriptors:
                activeDescriptors.remove("3dsc")
            else:
                activeDescriptors.append("3dsc")
        if choice == 3:
            return

def configureActiveObjectCounts():
    while True:
        run_menu = TerminalMenu([
            "Generate results for scene with 1 uncluttered object: " + ("enabled" if "1" in activeObjectCounts else "disabled"),
            "Generate results for scene with 4 added clutter objects: " + ("enabled" if "5" in activeObjectCounts else "disabled"),
            "Generate results for scene with 9 added clutter objects: " + ("enabled" if "10" in activeObjectCounts else "disabled"),
            "done"], title='-- Configure object counts to be tested --')
        choice = run_menu.show()
        if choice == 0:
            if "1" in activeObjectCounts:
                activeObjectCounts.remove("1")
            else:
                activeObjectCounts.append("1")
        if choice == 1:
            if "5" in activeObjectCounts:
                activeObjectCounts.remove("5")
            else:
                activeObjectCounts.append("5")
        if choice == 2:
            if "10" in activeObjectCounts:
                activeObjectCounts.remove("10")
            else:
                activeObjectCounts.append("10")
        if choice == 3:
            activeObjectCounts.sort()
            return

def configureSpinImageAngle():
    global spinImageSupportAngle
    spinangle_menu = TerminalMenu([
        "Set spin image support angle to 180 degrees (used for most charts)",
        "Set spin image support angle to 60 degrees (used for Figure 11)"],
        title='-- Configure spin image support angle to use during testing --')
    choice = spinangle_menu.show()
    if choice == 0:
        spinImageSupportAngle = 180
    if choice == 1:
        spinImageSupportAngle = 60

def configureGPU():
    global gpuID
    run_command_line_command('src/clutterbox/build/clutterbox --list-gpus')
    print()
    gpuID = input('Enter the ID of the GPU to use (usually 0): ')


def runClutterbox():
    while True:
        run_menu = TerminalMenu([
            "Run experiment with random seed drawn from seed list at random",
            "Run experiment with random seed with specific index in seed list",
            "Run experiment with manually entered random seed",
            "Configure descriptors to test (currently active: " + ', '.join(activeDescriptors) + ")",
            "Configure object counts (currently active: " + ', '.join(activeObjectCounts) + ")",
            "Configure Spin Image support angle (currently set to " + str(spinImageSupportAngle) + ")",
            "Configure GPU (use if system has more than one, currently set to GPU " + str(gpuID) + ")",
            "back"], title='------------ Run Clutterbox Experiment ------------')
        choice = run_menu.show()

        if choice == 0:
            chosenSeed = random.choice(random_seeds)
            executeClutterboxExperiment(chosenSeed)
            print()
        if choice == 1:
            chosenSeedIndex = input('Specify the index of the random seed to use (must be between 0 and ' + str(len(random_seeds)) + '): ')
            chosenSeed = random_seeds[int(chosenSeedIndex)]
            executeClutterboxExperiment(chosenSeed)
            print()
        if choice == 2:
            chosenSeed = input('Manually specify random seed to use (must be an integer!): ')
            executeClutterboxExperiment(chosenSeed)
            print()
        if choice == 3:
            configureActiveDescriptors()
            print()
        if choice == 4:
            configureActiveObjectCounts()
            print()
        if choice == 5:
            configureSpinImageAngle()
            print()
        if choice == 6:
            configureGPU()
            print()
        if choice == 7:
            return

def executeClutterEstimator(indexToCompute):
    global clutterFileMap
    if clutterFileMap is None:
        print()
        print('In order to be able to show which files must be compared to those generated,')
        print('the directory of clutter estimate files computed by the authors needs to be scanned.')
        print('This should only take a few moments.')
        clutterFileMap = {}

        clutterfiles = [name for name in os.listdir(estimatedClutterDirectory)
                     if os.path.isfile(os.path.join(estimatedClutterDirectory, name))]
        for fileindex, clutterfile in enumerate(clutterfiles):
            print('Processed', fileindex+1, 'of', len(clutterfiles), end='\r', flush=True)
            with open(os.path.join(estimatedClutterDirectory, clutterfile), 'r') as openFile:
                # Read JSON file
                try:
                    clutterFileContents = json.loads(openFile.read())
                    seed = clutterFileContents['sourceFile'].split('.')[0].split('_')[2]
                    clutterFileMap[seed] = os.path.join(estimatedClutterDirectory, clutterfile)
                except Exception as e:
                    print('FAILED TO READ FILE: ' + str(clutterfile))
                    print(e)
                    continue
        print()
        print()

    dumpfiles = [name for name in os.listdir(clutterSourceDumpFileDirectory)
                 if os.path.isfile(os.path.join(clutterSourceDumpFileDirectory, name))]
    dumpfiles.sort()

    print('Computing clutter estimate file for file index', indexToCompute)
    print()
    run_command_line_command('src/clutterbox/build/clutterEstimator '
                             '--result-dump-dir=' + clutterSourceDumpFileDirectory + ' '
                             '--object-dir=input/SHREC17/ '
                             '--output-dir=output/estimated_clutter '
                             '--compute-single-index=' + str(indexToCompute) + ' '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--samples-per-triangle=30 ')
    print()
    print('You should compare this produced file to:')
    print()
    dumpFilePath = os.path.join(clutterSourceDumpFileDirectory, dumpfiles[indexToCompute])
    try:
        with open(dumpFilePath, 'r') as openFile:
            dumpFileContents = json.loads(openFile.read())
            dumpFileSeed = dumpFileContents['seed']
            print('   ', clutterFileMap[str(dumpFileSeed)])
    except Exception as e:
        print('FAILED TO READ FILE: ' + dumpFilePath, e)
    print()

def runClutterEstimation():
    fileCount = len([name for name in os.listdir(clutterSourceDumpFileDirectory)
                     if os.path.isfile(os.path.join(clutterSourceDumpFileDirectory, name))])
    while True:
        clutterEstimation_menu = TerminalMenu([
            "Run Clutter Estimator with random file index",
            "Run Clutter Estimator with manually specified file index",
            "Configure GPU (use if system has more than one, currently set to GPU " + str(gpuID) + ")",
            "back"], title='------------ Run Clutter Estimator ------------')
        choice = clutterEstimation_menu.show()
        if choice == 0:
            index = random.randint(0, fileCount)
            executeClutterEstimator(index)
        if choice == 1:
            index = str(input('Specify index of file to process (must be an integer between 0 and ' + str(fileCount) + '): '))
            executeClutterEstimator(index)
        if choice == 2:
            configureGPU()
        if choice == 3:
            activeObjectCounts.sort()
            return

def runProjectionBenchmark():
    if ask_for_confirmation('This benchmark will consume around 30GB of RAM. You should close any applications to ensure you do not run out. Continue?'):
        print()
        print('Compiling..')
        os.makedirs('src/clutterbox/build', exist_ok=True)
        run_command_line_command('g++ -O3 -I ../lib/eigen/eigen3/ ../tools/projectionBenchmark/compare.cpp -o benchmark', 'src/clutterbox/build')
        print()
        print('Compilation complete, running benchmark..')
        print()
        run_command_line_command('benchmark', 'src/clutterbox/build')

def runOBJDump():
    global activeDescriptors
    global activeObjectCounts

    os.makedirs('output/highlightedobjects/figure15a', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure15b', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure15c', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure15d', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure15e', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure15f', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure16/toprank', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure16/top4ranks', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure16/top6ranks', exist_ok=True)
    os.makedirs('output/highlightedobjects/figure16/top12ranks', exist_ok=True)

    while True:
        visualisation_menu = TerminalMenu([
            "Generate OBJ files for Figure 15a",
            "Generate OBJ files for Figure 15b",
            "Generate OBJ files for Figure 15c",
            "Generate OBJ files for Figure 15d",
            "Generate OBJ files for Figure 15e",
            "Generate OBJ files for Figure 15f",
            "Generate OBJ files for Figure 16",
            "Configure descriptors to test (currently active: " + ', '.join(activeDescriptors) + ")",
            "Configure object counts (currently active: " + ', '.join(activeObjectCounts) + ")",
            "Configure Spin Image support angle (currently set to " + str(spinImageSupportAngle) + ")",
            "Configure GPU (use if system has more than one, currently set to GPU " + str(gpuID) + ")",
            "back"], title='------------ Dump OBJ files with match visualisation ------------')
        choice = visualisation_menu.show()
        if choice == 0:
            executeClutterboxExperiment('3056361425', 'output/highlightedobjects/figure15a', 0, 'output/highlightedobjects/figure15a')
        if choice == 1:
            executeClutterboxExperiment('3461184303', 'output/highlightedobjects/figure15b', 0, 'output/highlightedobjects/figure15b')
        if choice == 2:
            executeClutterboxExperiment('1919129218', 'output/highlightedobjects/figure15c', 0, 'output/highlightedobjects/figure15c')
        if choice == 3:
            executeClutterboxExperiment('3617347629', 'output/highlightedobjects/figure15d', 0, 'output/highlightedobjects/figure15d')
        if choice == 4:
            executeClutterboxExperiment('3500854400', 'output/highlightedobjects/figure15e', 0, 'output/highlightedobjects/figure15e')
        if choice == 5:
            executeClutterboxExperiment('3098714219', 'output/highlightedobjects/figure15f', 0, 'output/highlightedobjects/figure15f')
        if choice == 6:
            print('Overriding settings with those used to generate images..')
            print()
            backup_objectCounts = activeObjectCounts
            backup_descriptors = activeDescriptors
            activeObjectCounts = ['1']
            activeDescriptors = ['rici']

            print('Generating top rank visualisation')
            executeClutterboxExperiment('3048759171', 'output/highlightedobjects/figure16/toprank', 0)
            print()
            print('Generating top 4 ranks visualisation')
            executeClutterboxExperiment('3048759171', 'output/highlightedobjects/figure16/top4ranks', 3)
            print()
            print('Generating top 6 ranks visualisation')
            executeClutterboxExperiment('3048759171', 'output/highlightedobjects/figure16/top6ranks', 5)
            print()
            print('Generating top 12 ranks visualisation')
            executeClutterboxExperiment('3048759171', 'output/highlightedobjects/figure16/top12ranks', 11)
            print()
            activeObjectCounts = backup_objectCounts
            activeDescriptors = backup_descriptors
        if choice == 7:
            configureActiveDescriptors()
            print()
        if choice == 8:
            configureActiveObjectCounts()
            print()
        if choice == 9:
            configureSpinImageAngle()
            print()
        if choice == 10:
            configureGPU()
            print()
        if choice == 11:
            return

def runMainMenu():
    main_menu = TerminalMenu([
        "1. Install dependencies",
        "2. Download datasets",
        "3. Compile project",
        "4. Compile author generated results into spreadsheets",
        "5. Run Clutterbox experiment",
        "6. Run Clutter fraction estimation",
        "7. Run projection algorithm benchmark (Table 1)",
        "8. Dump result visualisation OBJ files",
        "9. exit"], title='---------------------- Main Menu ----------------------')

    while True:
        choice = main_menu.show()

        if choice == 0:
            installDependenciesMenu()
        if choice == 1:
            downloadDatasetsMenu()
        if choice == 2:
            compileProject()
        if choice == 3:
            runSpreadsheetBuilder()
        if choice == 4:
            runClutterbox()
        if choice == 5:
            runClutterEstimation()
        if choice == 6:
            runProjectionBenchmark()
        if choice == 7:
            runOBJDump()
        if choice == 8:
            return

def runIntroSequence():
    print()
    print('Greetings!')
    print()
    print('This script is intended to reproduce various figures in an interactive')
    print('(and hopefully convenient) manner.')
    print()
    print('It is recommended you refer to the included PDF manual for instructions')
    print()
    runMainMenu()


if __name__ == "__main__":
    runIntroSequence()