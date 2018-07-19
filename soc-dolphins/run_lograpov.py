#!/usr/bin/env python3

import sys
sys.path.insert(0, '../LoGraPov_v1.0')
import os
import numpy as np
import networkx as nx
import scipy.stats as stats
import time
from point_process import *
from graph_manip import *
from util import *


def main():
    if len(sys.argv) != 3:
        print("Error: wrong number of args")
        print("usage: ./run_lograpov.py task config_file")
        print("note: task can be rewire, hawkes, eclog, node_centric, or linear_regression")
        print("ex: ./run_lograpov.py rewire config.json")
        exit()
    
    task = sys.argv[1]
    config_file = sys.argv[2]
    
    if task == 'initialize':
        os.system("mkdir rewired && chmod u+rwx rewired")
        os.system("mkdir hawkes && chmod u+rwx hawkes")
        os.system("mkdir eclog && chmod u+rwx eclog")
        os.system("python3 ../LoGraPov_v1.0/initialize.py %s" % (config_file))

    elif task == 'rewire':
  	    os.system("qsub submit_rewiring.pbs")

    elif task == 'hawkes':
        os.system("python3 ../LoGraPov_v1.0/wr_bundle_sample_hawkes.py %s && qsub submit_bundle_sample_hawkes.pbs" % (config_file))

    elif task == 'eclog':
        os.system("python3 ../LoGraPov_v1.0/format_eclog_input.py %s && python3 ../LoGraPov_v1.0/wr_submit_eclog.py %s && python3 ../LoGraPov_v1.0/submit_mul_eclog.py %s" % (config_file, config_file, config_file))

    elif task == 'node_centric':
        os.system("qsub submit_edge2node.pbs")

    elif task == 'linear_regression':
        os.system("qsub submit_linear_regression.pbs")

    else:
        print("Error: unrecognized task.")
        print("task = rewire, hawkes, eclog, node_centric, or linear_model")
        exit()

    return

if __name__ == "__main__":
    main()
