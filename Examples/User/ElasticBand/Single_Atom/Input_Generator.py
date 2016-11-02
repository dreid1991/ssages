#! /usr/bin/env python

# Script for generating JSON input file for SSAGES FTS method from a template
# input JSON file with multiple walkers (one per node on string)

import json
import numpy as np
import copy

# Open template and load in the json data.
root = {}
with open('Template_Input.json') as f:
	root = json.load(f)

num = 16

centers_1 = np.linspace(-0.7, 0.7, num)
centers_2 = np.linspace(-0.5, 1.0, num)

# Add on the requested number of objects -1 because we are appending
for i in range(0,num - 1):
	root['driver'].append(copy.deepcopy(root['driver'][0]))

for i in range(num):
	# Change the log file name so each driver uses a different log file
	#root['driver'][i]['logfile'] = "log"

	# Change the node's location
        root['driver'][i]['method']['centers'][0] = round(centers_1[i], 3)
        root['driver'][i]['method']['centers'][1] = round(centers_2[i], 3)

# Convert python dictionary into JSON file
with open('Elastic.json', 'w') as f:
		json.dump(root, f, indent=4, separators=(',', ': '))
