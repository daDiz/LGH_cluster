#!/usr/bin/env python3

import sys
sys.path.insert(0, './')

import numpy as np
import json
from util import *

def main():
	if len(sys.argv) != 2:
		print("Error: not enough args")
		print("usage: python3 wr_bundle_list.py config")
		exit()

	config_file = sys.argv[1]

	config = parse_config(config_file)

	input_dir = config['hawkes']['input_dir']
	graph_name = config['hawkes']['src_graph']
	num_graphs = config['hawkes']['num_graphs']
	max_lamb = config['hawkes']['max_lamb']
	alpha = config['hawkes']['alpha']
	num_samples = config['hawkes']['num_samples']
	distinct = config['hawkes']['distinct']


	for i in range(0, num_graphs):
		with open("%s/hawkes/sample_hawkes_config%d.json" % (input_dir, i), 'w') as file:
			content = {
			"src_graph": "%s/rewired/%s_rewired%d.txt" % (input_dir, graph_name, i), 
			"max_lamb": max_lamb,
			"alpha": alpha,
			"num_samples": num_samples,
			"distinct": distinct,
			"output": "%s/hawkes/sample_hawkes%d.txt" % (input_dir, i)
			}
			json.dump(content, file)

	with open('%s/sample_hawkes_list.txt' % (input_dir), 'w') as file:
		for i in range(0, num_graphs):
			file.write("~/LoGraPov_v1.0/sample_hawkes.py hawkes/sample_hawkes_config%d.json > hawkes/sample_hawkes%d.log\n" % (i, i))

	return

if __name__ == "__main__":
	main()