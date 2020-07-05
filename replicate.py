from simple_term_menu import TerminalMenu

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download SHREC 2017 3D shape dataset (~10GB download, ~50GB extracted)",
        "Download experiment results generated by authors (~25GB download, ~40GB extracted)",
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show()

        if choice == 0:
            print('Downloading SHREC 2017 dataset..')
            print()
        if choice == 1:
            print('Downloading author generated results')
            print()
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
            run_command_line_command('sudo apt install cmake python3 python3-pip libpcl-dev g++-7 gcc-7')
            run_command_line_command('sudo pip3 install console-menu')
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

    compileProjectMenu = TerminalMenu([
        "Run cmake (must run before make)",
        "Run make",
        "back"], title='------------------- Compile Project -------------------')

    while True:
        choice = compileProjectMenu.show()

        if choice == 0:
            run_command_line_command('cmake ..', 'src/clutterbox/build')
        if choice == 1:
            run_command_line_command('make -j 4', 'src/clutterbox/build')
        if choice == 2:
            return

def executeClutterboxExperiment():
    run_command_line_command('clutterbox '
                             '--box-size=1 '
                             '--source-directory=../../../input/SHREC17/ '
                             '--object-counts=1,5,10 '
                             '--override-total-object-count=10 '
                             '--descriptors=rici,si,3dsc '
                             '--support-radius=0.3 '
                             '--force-gpu=0 '
                             '--force-seed=0 '
                             '--spin-image-support-angle-degrees=180 '
                             '--3dsc-min-support-radius=0.048 '
                             '--3dsc-point-density-radius=0.096 '
                             '--dump-raw-search-results')

def runClutterbox():
    while True:
        run_menu = TerminalMenu([
            "Run seed drawn from seed list at random",
            "Run seed with specific index in seed list",
            "Configure descriptors to test",
            "Configure object counts",
            "Configure Spin Image support angle",
            "Configure GPU (use if system has more than one)",
            "back"], title='------------ Run Clutterbox Experiment -----------')
        choice = run_menu.show()

        if choice == 0:
            executeClutterboxExperiment()
            print()
        if choice == 1:
            run_command_line_command('sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev')
            print()
        if choice == 2:
            return

def runMainMenu():
    main_menu = TerminalMenu([
        "1. Install dependencies",
        "2. Download datasets",
        "3. Compile project",
        "4. Run Clutterbox experiment",
        "5. Compile author generated results into spreadsheets",
        "6. Run projection algorithm benchmark (Table 1)",
        "7. exit"], title='---------------------- Main Menu ----------------------')

    while True:
        choice = main_menu.show()

        if choice == 0:
            installDependenciesMenu()
        if choice == 1:
            downloadDatasetsMenu()
        if choice == 2:
            compileProject()
        if choice == 3:
            runClutterbox()
        if choice == 6:
            return

def runIntroSequence():
    print()
    print('Greetings!')
    print()
    print('This script is intended to reproduce various figures in an interactive')
    print('(and hopefully convenient) manner.')
    print()
    print('However, a set of bash scripts is provided in the "scriptsrc" folder')
    print('that will allow you to circumvent parts of this script, if desired.')
    print()
    runMainMenu()


if __name__ == "__main__":
    runIntroSequence()