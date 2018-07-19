import numpy as np
import networkx as nx
import scipy.stats as stats
from point_process import *
from graph_manip import *
import sys

def let_hawkes_fly(target, alpha, maxgen=100):
    G = nx.read_edgelist('%s/%s.txt' % (target, target), nodetype=int)
    max_eig = lambda_max(G)
    theta = alpha / max_eig
    print('max eigen value: %f' % (max_eig))
    print('theta: %f' % (theta))
    
    #T = exact_hawkes_maxgen_all(G, maxgen, theta)   
    #T = exact_hawkes_all(G, theta)
    #nodes = list(G.nodes)

    #df = pd.DataFrame(nodes, columns=['node'])
    #df['hawkes'] = T

    #df = df.sort_values(by='hawkes', ascending=False)
    #df.to_csv('%s/rank_node_hawkes.txt' % (target), header=False, index=False, sep=' ')

if __name__ == "__main__":
    if len(sys.argv) != 4:
       print("Error: not enough args")
       print("usage: python3 single_hawkes.py target alpha maxgen")
       print("ex: python3 single_hawkes.py socfb-Swarthmore42 0.99")
       exit()

    target = sys.argv[1]
    alpha = np.float(sys.argv[2])
    maxgen = np.int(sys.argv[3])

    let_hawkes_fly(target, alpha, maxgen)
