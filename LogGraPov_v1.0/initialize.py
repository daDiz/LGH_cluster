#!/usr/bin/env python3

import sys
sys.path.insert(0, './')

import numpy as np 
from util import *

def main():
	if len(sys.argv) != 2:
		print("Error: wrong number of args")
		print("usage: python3 wr_submit_rewire.py config.json")
		exit()

	config_file = sys.argv[1]
	config = parse_config(config_file)
	
	## create submit_rewiring.pbs
	with open('submit_rewiring.pbs', 'w') as file:
		line = "#!/bin/bash\n" +  \
			"#PBS -l nodes=%s:ppn=%s\n" % (config['initialize']['rewire']['nodes'], config['initialize']['rewire']['ppn']) + \
			"#PBS -l walltime=%s\n" % (config['initialize']['rewire']['walltime']) + \
			"#PBS -N rewire_graph\n" + \
			"#PBS -q cpu\n" + \
			"#PBS -V\n\n" + \
			"cd $PBS_O_WORKDIR\n" + \
			"aprun -n %s python3 ~/LoGraPov_v1.0/rewire.py %s > rewire.log" % (config['initialize']['rewire']['nodes'], config_file)

		file.write(line)

	## create submit_bundle_sample_hawkes.pbs
	with open('submit_bundle_sample_hawkes.pbs', 'w') as file:
		line = "#!/bin/bash\n" +  \
			"#PBS -l nodes=%s:ppn=%s\n" % (config['initialize']['hawkes']['nodes'], config['initialize']['hawkes']['ppn']) + \
			"#PBS -l walltime=%s\n" % (config['initialize']['hawkes']['walltime']) + \
			"#PBS -N sample_hawkes\n" + \
			"#PBS -q cpu\n" + \
			"#PBS -V\n\n" + \
			"cd $PBS_O_WORKDIR\n" + \
			"aprun -n %s pcp sample_hawkes_list.txt" % (config['initialize']['hawkes']['num_cores'])

		file.write(line)

	## create submit_edge2node.pbs
	with open('submit_edge2node.pbs', 'w') as file:
		line = "#!/bin/bash\n" +  \
			"#PBS -l nodes=%s:ppn=%s\n" % (config['initialize']['node_centric']['nodes'], config['initialize']['node_centric']['ppn']) + \
			"#PBS -l walltime=%s\n" % (config['initialize']['node_centric']['walltime']) + \
			"#PBS -N edge2node\n" + \
			"#PBS -q cpu\n" + \
			"#PBS -V\n\n" + \
			"cd $PBS_O_WORKDIR\n" + \
			"aprun -n %s python3 ~/LoGraPov_v1.0/edge2node_eclog.py %s > edge2node.log" % (config['initialize']['node_centric']['nodes'], config_file)

		file.write(line)

	## create submit_linear_regression.pbs
	with open('submit_linear_regression.pbs', 'w') as file:
		line = "#!/bin/bash\n" +  \
			"#PBS -l nodes=%s:ppn=%s\n" % (config['initialize']['linear_regression']['nodes'], config['initialize']['node_centric']['ppn']) + \
			"#PBS -l walltime=%s\n" % (config['initialize']['linear_regression']['walltime']) + \
			"#PBS -N linear_regression\n" + \
			"#PBS -q cpu\n" + \
			"#PBS -V\n\n" + \
			"cd $PBS_O_WORKDIR\n" + \
			"aprun -n %s python3 ~/LoGraPov_v1.0/build_linear_model.py %s > linear_regression.log" % (config['initialize']['linear_regression']['nodes'], config_file)

		file.write(line)

	return

if __name__ == "__main__":
	main()
