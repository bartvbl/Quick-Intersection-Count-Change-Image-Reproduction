import json
import os
import os.path

resultDirectory = '../../output/HEIDRUNS/output_majorfix_v2_earlycutoff/output'
outfile = 'seeds_earlycutoff.txt'

originalFiles = os.listdir(resultDirectory)

with open(outfile, 'w') as outFile:
	for fileindex, file in enumerate(originalFiles):
		print(str(fileindex+1) + '/' + str(len(originalFiles)), file, end='\r')
		if(file == 'raw'):
			continue
		with open(os.path.join(resultDirectory, file), 'r') as openFile:
			fileContents = json.loads(openFile.read())
			outFile.write(str(fileContents['seed']) + '\n')