import os
import os.path
import json

def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    results = []
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex+1) + '/' + str(len(originalFiles)), file)
        if(file == 'raw'):
        	continue
        with open(os.path.join(path, file), 'r') as openFile:
            fileContents = json.loads(openFile.read())
            results.append(fileContents)
    return results

results = loadOutputFileDirectory('../input/hamming_tree_query_times_by_authors/')

counts = [0] * 4096
sums = [0] * 4096

with open('../output/figure_7_query_times.csv', 'w') as outFile:
    outFile.write('Index, End time\n') # , Counts
    for i in range(0, len(results)):
        for row in range(0, 4096):
            value = results[i]['indexedQueryResults']['distanceTimes'][row]
            if value != -1:
                counts[row] += 1
                sums[row] += value
    for i in range(0, 4096):
        if counts[i] == 0:
            continue
        outFile.write(str(i) + ', ' + str(float(sums[i]) / float(counts[i])) + '\n') #  + ', ' + str(counts[i])