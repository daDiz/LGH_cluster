import numpy as np
import pandas as pd
import networkx as nx
from graph_manip import *
import sys

def check_lamb(src):
    g = nx.read_edgelist('%s/%s.txt' % (src, src), nodetype=int)
    #d = mean_degree_2nd(g)
    m = degree_max2(g)
    #n = len(g.nodes)
    print("max degree: %d" % (m))
    print("1 / max_degree = %f" % (1.0/m))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: not enough args")
        print("python3 check_lambda.py src")
        print("python3 check_lambda.py socfb-B-anon")
    
    src = sys.argv[1]
    check_lamb(src)


