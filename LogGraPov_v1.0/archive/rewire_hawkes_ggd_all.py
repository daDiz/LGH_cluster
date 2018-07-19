import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from point_process import *
from graph_manip import *

##################################
# parameters
####################################

## karate
#src_graph = 'karate/karate_1.txt'
dest_path = 'karate/karate_rewired_all/karate'
#outfile = 'karate/karate_all_exactHawkes_labeled.txt'
#alpha = 0.96 # theta / lambda_max


## soc-dolphins
#src_graph = 'soc-dolphins/soc-dolphins.txt'
#dest_path = 'soc-dolphins/dolphins_rewired_all/soc-dolphins'
#outfile = 'soc-dolphins/soc-dolphins_all_exactHawkes_labeled.txt'
#alpha = 0.95 # theta / lambda_max

## rt-retweet
#src_graph = 'rt-retweet/rt-retweet.txt'
#dest_path = 'rt-retweet/rt-retweet_rewired_all/rt-retweet'
#outfile = 'rt-retweet/rt-retweet_all_exactHawkes_labeled.txt'
#alpha = 0.98 # theta / lambda_max

###########################################
#nswap = 10
num_graphs = 1000


##############################################
#G = nx.read_edgelist(src_graph)
#theta = alpha * 1. / lambda_max(G)

#num_nodes = len(G.nodes)
#num_edges = len(G.edges)

#os.system('echo rewiring...')
#TE, node_list = gen_dpr_exact_all(G, nswap, num_graphs, theta, path=dest_path)
#os.system('echo finish rewiring')

# generate input for e-clog
for i in range(0, num_graphs):
    G1 = nx.read_edgelist(dest_path + '_rewired%d.txt' % (i))
    edges = list(G1.edges)
    with open(dest_path + '_graphlets_rewired%d.txt' % (i), 'w') as file:
        file.write('%d %d\n' % (num_nodes, num_edges))
        for e in edges:
            x = np.int(e[0])-1
            y = np.int(e[1])-1
            file.write('%s %s\n' % (x,y))


os.system('echo start hawkes')
os.system('./mul_eclog.sh %d %s' % (num_graphs-1, dest_path))
os.system('echo finish hawkes')



###############################################
# convert edge-centric gfd to node-centric gfd
###############################################
X = []
for i in range(0, num_graphs):
    df = pd.read_csv(dest_path + '_graphlets_rewired%d.txt_local_graphlet_freqeuncy_5_omp.txt' % (i), header=None, sep=' |: ')
    E = df.values
    for j in node_list[i]:
        X.append(list(sum_lgd(j-1, E)))

X = np.array(X)


df = pd.DataFrame(X)
df[46] = TE.flatten()
df[47] = node_list.flatten()
df.to_csv(outfile, header=False, index=False, sep=' ')

