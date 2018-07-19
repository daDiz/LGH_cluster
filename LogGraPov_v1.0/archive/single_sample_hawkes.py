#!/usr/bin/env python3

import sys
sys.path.insert(0, '../')
import numpy as np
import networkx as nx
import scipy.stats as stats

from point_process import *
from graph_manip import *

def let_hawkes_fly(target, alpha, Nsamples, distinct):
    G = nx.read_edgelist('%s.txt' % (target), nodetype=int)

    theta = alpha / lambda_max(G)

    T = sample_hawkes_all(G, Nsamples, theta, distinct)

    
    nodes = list(G.nodes)

    df = pd.DataFrame(nodes, columns=['node'])
    df['hawkes'] = T

    df.to_csv('sample_hawkes.txt', header=False, index=False, sep=' ')
    #df = df.sort_values(by='hawkes', ascending=False)
    #df.to_csv('%s/rank_node_sample_hawkes.txt' % (target), header=False, index=False, sep=' ')

if __name__ == "__main__":
    if len(sys.argv) != 5:
       print("Error: not enough args")
       print("usage: python3 single_sample_hawkes.py target alpha Nsamples distinct")
       print("ex: python3 single_sample_hawkes.py socfb-Swarthmore42 0.99 1 True")
       exit()

    target = sys.argv[1]
    alpha = np.float(sys.argv[2])
    Nsamples = np.int(sys.argv[3])
    distinct = np.bool(sys.argv[4])

    let_hawkes_fly(target, alpha, Nsamples, distinct)
