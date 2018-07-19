#!/usr/bin/env python3

import sys
sys.path.insert(0, './')

import numpy as np 
from graph_manip import *
from util import *

def main():
	if len(sys.argv) != 2:
		print("Error: not enough args")
		print("usage: python3 format_eclog_input.py config")
		print('ex: python3 format_eclog_input.py config.json')
		exit()
		
	config_file = sys.argv[1]
	config = parse_config(config_file)

	src_path = config['eclog']['format_input']
	num_graphs = config['eclog']['num_graphs']

	gen_eclog_input(src_path, num_graphs)

	return 

if __name__=="__main__":
	main()