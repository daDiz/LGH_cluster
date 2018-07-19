#!/usr/bin/env python3

import sys
sys.path.insert(0, './')
import numpy as np
import networkx as nx
import time
from util import *
from graph_manip import *



def main():
	if len(sys.argv) != 2:
		print("Error: not enough args")
		print("usage: python3 rewire.py config_file")
		exit()
	
	config_file = sys.argv[1]
	
	config = parse_config(config_file)

	src_graph = config['rewire']['src_graph']
	nswap = config['rewire']['nswap']
	max_tries = config['rewire']['max_tries']
	num_graphs = config['rewire']['num_graphs']
	dest_path = config['rewire']['dest_path']

	
	G = nx.read_edgelist(src_graph)

	dpr_rewire(G, nswap, max_tries, num_graphs, path=dest_path)

	return


if __name__ == "__main__":
	start_time = time.time()
	main()
	print("rewiring time %s s" % (time.time() - start_time))
