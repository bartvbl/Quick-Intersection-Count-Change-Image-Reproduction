import subprocess
import os.path
import os

seedFilePath = "seeds_lotsofobjects.txt"
outputDir = '../../output/sceneOBJs/'

allSeeds = []
with open(seedFilePath) as seedFile:
    allSeeds = [line.strip() for line in seedFile]

for seed in allSeeds:
    print('Processing seed', seed)
    subprocess.run(['../../cmake-build-release/riciverification --source-directory="/home/bart/Datasets/SHREC2017/" --box-size=1 --object-counts=1,5,10 --support-radius=0.3 --3dsc-min-support-radius=0.048 --3dsc-point-density-radius=0.096 --spin-image-support-angle-degrees=180 --dump-raw-search-results --override-total-object-count=10 --descriptors=none --scene-obj-file-dump-directory="' + outputDir + '" --force-seed=' + str(seed)], shell=True)

    objFile_1 = os.path.abspath(os.path.join(outputDir, str(seed) + '_1.obj'))
    objFile_5 = os.path.abspath(os.path.join(outputDir, str(seed) + '_5.obj'))
    objFile_10 = os.path.abspath(os.path.join(outputDir, str(seed) + '_10.obj'))

    subprocess.run(['../../../.external/meshlab/build/distrib/meshlab ' + objFile_10], shell=True, cwd="../../../.external/meshlab/build")

    print()
    answer = input('Should this model be kept? y/n: ')
    if answer == 'y':
        subprocess.run(['../../../.external/meshlab/build/distrib/meshlab ' + objFile_1], shell=True, cwd="../../../.external/meshlab/build")
        subprocess.run(['../../../.external/meshlab/build/distrib/meshlab ' + objFile_5], shell=True, cwd="../../../.external/meshlab/build")
    else:
        os.remove(objFile_1)
        os.remove(objFile_5)
        os.remove(objFile_10)
