import numpy as np
import networkx as nx
import scipy.stats as stats
from point_process import *
from graph_manip import *
import sys

def let_hawkes_fly(target, alpha, Nsamples=1):
    G = nx.read_edgelist('%s/%s.txt' % (target, target), nodetype=int)

    theta = alpha / lambda_max(G)
    N = len(G.nodes)
    E = []
    for n1, n2 in G.edges:
        E.append([n1, n2])
        E.append([n2, n1])
    E = np.array(E)

    T = sample_hawkes_fly(E, N, Nsamples, theta)
    
    nodes = list(range(1, N+1))

    df = pd.DataFrame(nodes, columns=['node'])
    df['hawkes'] = T

    df = df.sort_values(by='hawkes', ascending=False)
    df.to_csv('%s/rank_node_sample_hawkes.txt' % (target), header=False, index=False, sep=' ')

if __name__ == "__main__":
    if len(sys.argv) != 4:
       print("Error: not enough args")
       print("usage: python3 single_sample_hawkes.py target alpha Nsamples")
       print("ex: python3 single_sample_hawkes.py socfb-Swarthmore42 0.99 1")
       exit()

    target = sys.argv[1]
    alpha = np.float(sys.argv[2])
    Nsamples = np.int(sys.argv[3])

    let_hawkes_fly(target, alpha, Nsamples)
