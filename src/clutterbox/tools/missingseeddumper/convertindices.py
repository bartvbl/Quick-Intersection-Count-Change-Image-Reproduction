import sys

with open(sys.argv[1], 'r') as f:
	indices = f.readlines()
with open(sys.argv[2], 'r') as f:
	seeds = f.readlines()

pickedSeeds = [seeds[int(x)] for x in indices]

with open(sys.argv[3], 'w') as f:
	f.write(''.join(pickedSeeds))