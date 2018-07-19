#!/usr/bin/env python3

import sys
sys.path.insert(0, './')

import numpy as np 
from util import *

def main():
	if len(sys.argv) != 2:
		print("Error: wrong number of args")
		print("usage: python3 wr_submit_eclog.py config.json")
		exit()

	config_file = sys.argv[1]
	config = parse_config(config_file)

	src_graph = config['eclog']['src_graph']
	num_graphs = config['eclog']['num_graphs']
	num_threads = config['eclog']['num_threads']
	eclog_mode = config['eclog']['eclog_mode']
	output_dir = config['eclog']['output_dir']

	for i in range(0, num_graphs):
		with open('%s/submit_eclog%d.pbs' % (output_dir, i), 'w') as file:
			line = "#!/bin/bash\n" +  \
				"#PBS -l nodes=1:ppn=32\n" + \
				"#PBS -l walltime=10:00:00\n" + \
				"#PBS -N eclog\n" + \
				"#PBS -q cpu\n" + \
				"#PBS -V\n\n" + \
				"cd %s\n" % (output_dir) + \
				"export OMP_NUM_THREAD=%d\n" % (num_threads) + \
				"aprun -n 1 -d %d ~/E-CLoG_code/omp_graphlet_count_5 -i %s_graphlets_rewired%d.txt -t %d -%s > eclog%d.log" % (num_threads, src_graph, i, num_threads, eclog_mode, i)

			file.write(line)

	return

if __name__=="__main__":
	main()
