import json
import os
import subprocess
import random

from scripts.simple_term_menu import TerminalMenu

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)
    subprocess.run(command, shell=True, check=False, cwd=working_directory)

def ask_for_confirmation(message):
    confirmation_menu = TerminalMenu(["yes", "no"], title=message)
    choice = confirmation_menu.show()
    return choice == 0

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download SHREC 2017 3D shape dataset (7.9GB download, ~52.5GB extracted, needed for all Figures)",
        "Download experiment results generated by authors (~1.6GB download, 13.8GB extracted, needed for Figures 8-11)",
        "Download distance function values computed by authors (8.0GB download, 51.2GB extracted, needed for Figure 13)",
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
            if not os.path.isfile('input/download/results_computed_by_authors.7z') or ask_for_confirmation('It appears the results archive file has already been downloaded. Would you like to download it again?'):
                print('Downloading results archive file..')
                run_command_line_command('wget --output-document results_computed_by_authors.7z https://data.mendeley.com/datasets/p7g8fz82rk/1/files/29a722cc-b7b5-456a-a096-5d8ac55d6881/results_computed_by_authors.7z?dl=1', 'input/download/')
            print()
            run_command_line_command('p7zip -k -d download/results_computed_by_authors.7z', 'input/')

            print()
            if not os.path.isfile('input/download/results_computed_by_authors_quicci_fpfh.7z') or ask_for_confirmation('It appears the second results archive file has already been downloaded. Would you like to download it again?'):
                print('Downloading results archive file..')
                run_command_line_command('wget --output-document results_computed_by_authors_quicci_fpfh.7z https://data.mendeley.com/datasets/k9j5ymry29/2/files/50c14dc1-0514-491b-8aa1-d3a575f722c7/results_computed_by_authors_quicci_fpfh.7z?dl=1', 'input/download/')
            print()
            run_command_line_command('p7zip -k -d download/results_computed_by_authors_quicci_fpfh.7z', 'input/')

            print()
            if not os.path.isfile('input/download/clutter_estimated_by_authors.7z') or ask_for_confirmation('It appears the clutter estimates file has already been downloaded. Would you like to download it again?'):
                print('Downloading clutter estimates file..')
                run_command_line_command('wget --output-document clutter_estimated_by_authors.7z https://data.mendeley.com/datasets/p7g8fz82rk/1/files/37d353c5-7fd4-4488-a94a-97bb58dc722d/clutter_estimated_by_authors.7z?dl=1', 'input/download/')
            print()
            run_command_line_command('p7zip -k -d download/clutter_estimated_by_authors.7z', 'input/')

            print()
            print('Download and extraction complete. You may now delete the following files if you need the disk space:')
            print('- input/download/results_computed_by_authors.7z')
            print('- input/download/results_computed_by_authors_quicci_fpfh.7z')
            print('- input/download/clutter_estimated_by_authors.7z')
            print()

        if choice == 2:
            if not os.path.isfile('input/download/distances_computed_by_authors.7z') or ask_for_confirmation('It appears the computed distances archive file has already been downloaded. Would you like to download it again?'):
                print('Downloading distance function distances computed by authors..')
                run_command_line_command('wget --output-document distances_computed_by_authors.7z https://data.mendeley.com/datasets/k9j5ymry29/2/files/b3fe4f65-bf36-4fa7-9d26-217a59e35e54/distances_computed_by_authors.7z?dl=1', 'input/download/')
            print()
            os.makedirs('input/SHREC17', exist_ok=True)
            run_command_line_command('p7zip -k -d download/distances_computed_by_authors.7z', 'input/')
            print('Download and extraction complete. You may now delete the file input/download/distances_computed_by_authors.7z if you need the disk space.')
            print()

        if choice == 3:
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
            run_command_line_command('sudo pip3 install simple-term-menu xlwt xlrd numpy matplotlib pillow PyQt5')
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
            run_command_line_command('rm src/clutterbox/build/*')
            run_command_line_command('cmake ..', 'src/clutterbox/build')
        if choice == 1:
            run_command_line_command('make -j 4', 'src/clutterbox/build')
        if choice == 2:
            return

def runSpreadsheetBuilder():
    if ask_for_confirmation('The spreadsheet construction script will consume around 9GB of RAM.\nYou should close any applications to ensure you do not run out of memory.\nContinue?'):
        print()
        run_command_line_command('python3 compileresultfiles.py', 'scripts/')
        print()
        run_command_line_command('python3 generatechartsspreadsheet.py', 'scripts/')
        print()


activeDescriptors = ['rici', 'si', '3dsc', 'quicci', 'fpfh']
activeObjectCounts = ['1', '5', '10']
gpuID = 0
seedFileLocation = 'res/seeds_used_for_clutterbox_experiments.txt'
with open(seedFileLocation) as seedFile:
    random_seeds = [line.strip() for line in seedFile]

clutterSourceDumpFileDirectory = 'input/results_computed_by_authors/HEIDRUNS/output_qsifix_v4_withearlyexit/output'
estimatedClutterDirectory = 'input/clutter_estimated_by_authors/clutter/'

with open('output/filemap.json') as fileMapFile:
    clutterFileMap = json.loads(fileMapFile.read())

clutterSeedToFileMap = None



def executeClutterboxExperiment(randomSeed, matchVisualisationDirectory = None, matchVisualisationThreshold = 0, sceneOBJDumpDirectory = None):
    os.makedirs('output/clutterbox_results', exist_ok=True)
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
                             '--spin-image-support-angle-degrees=180 '
                             '--3dsc-min-support-radius=0.048 '
                             '--3dsc-point-density-radius=0.096 '
                             '--dump-raw-search-results ' +
                             visualisationParameters +
                             sceneOBJParameter)

    print('For each of the histograms in this output file, you should')
    print('compare them against the following dump files generated by the authors:')
    print()
    for descriptor in activeDescriptors:
        for objectCount in activeObjectCounts:
            print('- ' + descriptor.upper() + " descriptor, " + str(objectCount) + " objects in total in the clutterbox: ", end='')
            if descriptor == 'si':
                    descriptor += '180'
            if randomSeed in clutterFileMap[descriptor.upper()][str(objectCount)]:
                filePath = clutterFileMap[descriptor.upper()][str(objectCount)][str(randomSeed)]
                relativePath = os.path.relpath(os.path.abspath(os.path.normpath(os.path.join('scripts/', filePath))))
                print(relativePath)
            else:
                print('(missing)')
    print()


def configureActiveDescriptors():
    while True:
        run_menu = TerminalMenu([
            "Generate results for Radial Intersection Count Image: " + ("enabled" if "rici" in activeDescriptors else "disabled"),
            "Generate results for Spin Image: " + ("enabled" if "si" in activeDescriptors else "disabled"),
            "Generate results for 3D Shape Context: " + ("enabled" if "3dsc" in activeDescriptors else "disabled"),
            "Generate results for Quick Intersection Count Change Image: " + ("enabled" if "quicci" in activeDescriptors else "disabled"),
            "Generate results for Fast Point Feature Histograms: " + ("enabled" if "fpfh" in activeDescriptors else "disabled"),
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
            if "quicci" in activeDescriptors:
                activeDescriptors.remove("quicci")
            else:
                activeDescriptors.append("quicci")
        if choice == 4:
            if "fpfh" in activeDescriptors:
                activeDescriptors.remove("fpfh")
            else:
                activeDescriptors.append("fpfh")
        if choice == 5:
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
            configureGPU()
            print()
        if choice == 6:
            return

def executeClutterEstimator(indexToCompute):
    global clutterSeedToFileMap
    os.makedirs('output/estimated_clutter', exist_ok=True)
    if clutterSeedToFileMap is None:
        print()
        print('In order to be able to show which files must be compared to those generated,')
        print('the directory of clutter estimate files computed by the authors needs to be scanned.')
        print('This should only take a few moments.')
        clutterSeedToFileMap = {}

        clutterfiles = [name for name in os.listdir(estimatedClutterDirectory)
                     if os.path.isfile(os.path.join(estimatedClutterDirectory, name))]
        for fileindex, clutterfile in enumerate(clutterfiles):
            print('Processed', fileindex+1, 'of', len(clutterfiles), end='\r', flush=True)
            with open(os.path.join(estimatedClutterDirectory, clutterfile), 'r') as openFile:
                # Read JSON file
                try:
                    clutterFileContents = json.loads(openFile.read())
                    seed = clutterFileContents['sourceFile'].split('.')[0].split('_')[2]
                    clutterSeedToFileMap[seed] = os.path.join(estimatedClutterDirectory, clutterfile)
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
            print('   ', clutterSeedToFileMap[str(dumpFileSeed)])
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

def runSofaScene():
    os.makedirs('output/figure12/', exist_ok=True)
    run_command_line_command('src/clutterbox/build/quicciDistanceFunctionBenchmark '
                             '--source-directory=input/SHREC17/ '
                             '--force-seed=3456690118 '
                             '--support-radius=0.3 '
                             '--output-directory=output/figure12/ '
                             '--sphere-counts=500 '
                             '--experiment-mode=similar '
                             '--clutter-sphere-radius=0.05 '
                             '--scene-sphere-count=500 '
                             '--enable-obj-dump')
    print()
    print('You can find the produced OBJ file here:')
    print('    output/figure12/scene_similar_3456690118_500_spheres.obj')
    print()

def runDistanceFunctionEvaluation():
    while True:
        distanceFunctionEvaluation_menu = TerminalMenu([
            "Compute baseline result selected at random",
            "Compute baseline result with specific index",
            "Compute similar result selected at random",
            "Compute similar result with specific index",
            "Compile results computed by authors into heatmaps and charts shown in Figure 13",
            "back"], title='------------ Run Distance Functions Evaluation ------------')
        choice = distanceFunctionEvaluation_menu.show()
        if choice == 4:
            run_command_line_command('python3 buildDistanceFunctionsSpreadsheet.py', 'scripts/')
        if choice == 3:
            return

def runMainMenu():
    main_menu = TerminalMenu([
        "1. Install dependencies",
        "2. Download datasets",
        "3. Compile project",
        "4. Run Hamming Tree Evaluation (Figure 7)",
        "5. Compile author generated results into spreadsheets (Figures 8-11)",
        "6. Run Clutterbox experiment (used for Figures 8-11)",
        "7. Run Clutter fraction estimation (used for Figure 9)",
        "8. Render scene shown in Figure 12",
        "9. Run Evaluation of Distance Functions (Figure 13)",
        "10. exit"], title='---------------------- Main Menu ----------------------')

    while True:
        choice = main_menu.show()

        if choice == 0:
            installDependenciesMenu()
        if choice == 1:
            downloadDatasetsMenu()
        if choice == 2:
            compileProject()
        if choice == 3:
            pass
        if choice == 4:
            runSpreadsheetBuilder()
        if choice == 5:
            runClutterbox()
        if choice == 6:
            runClutterEstimation()
        if choice == 7:
            runSofaScene()
        if choice == 8:
            runDistanceFunctionEvaluation()
        if choice == 9:
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