#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, './')
import numpy as np 
from util import *

def main():
	if len(sys.argv) != 2:
		print("Error: wrong number of args")
		print("usage: ./submit_mul_eclog.py config")
		exit()

	config_file = sys.argv[1]
	config = parse_config(config_file)

	num_graphs = config['eclog']['num_graphs']
	output_dir = config['eclog']['output_dir']

	for i in range(0, num_graphs):
		os.system("qsub %s/submit_eclog%d.pbs" % (output_dir, i))

	return

if __name__=="__main__":
	main()
	