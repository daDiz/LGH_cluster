#!/usr/bin/env python3

import sys
sys.path.insert(0, './')
import numpy as np
import networkx as nx
import scipy.stats as stats
import time
from point_process import *
from graph_manip import *
from util import *


def main():
  if len(sys.argv) != 2:
    print("Error: not enough args")
    print("usage: python3 exact_hawkes.py config_file")
    print("ex: python3 exact_hawkes.py exact_hawkes_config.json")
    exit()
  
  config = sys.argv[1]
  
  src_graph, max_lamb, maxgen, alpha, output = parse_config_exact_hawkes(config)

  G = nx.read_edgelist(src_graph, nodetype=int)

  if max_lamb is None:
    print('Max eigen-value is not given.\nCalculate it from input graph. ')
    theta = alpha / lambda_max(G)
  else:
    theta = alpha / max_lamb

  T = exact_hawkes_maxgen_all(G, maxgen, theta)

  nodes = list(G.nodes)

  df = pd.DataFrame(nodes, columns=['node'])
  df['hawkes'] = T

  df.to_csv(output, header=False, index=False, sep=' ')
  
  return 

if __name__ == "__main__":
  start_time = time.time()
  main()
  print("exact hawkes finished in %s s" % (time.time() - start_time))



