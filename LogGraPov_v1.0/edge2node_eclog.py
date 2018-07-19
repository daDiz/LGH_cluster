#!/usr/bin/env python3

import sys
sys.path.insert(0, './')
import numpy as np
import networkx as nx
from graph_manip import *
from util import *


def main():
    if len(sys.argv) != 2:
        print("Error: wrong number of args")
        print("usage: python3 edge2node_eclog.py config")
        print("ex: python3 edge2node_eclog.py config.json")
        exit()

    config_file = sys.argv[1]
    config = parse_config(config_file)
    src_path = config['node_centric']['src_path']
    num_graphs = config['node_centric']['num_graphs']
    eclog_mode = config['node_centric']['eclog_mode']
    

    edge2node(src_path, num_graphs, eclog_mode)


if __name__ == "__main__":
    main()
