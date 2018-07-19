import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import networkx as nx
import os
import time
from point_process import *
from graph_manip import *

##################################
# parameters
####################################

## karate
#src_graph = 'soc-karate/soc-karate.txt'
#dest_path = 'soc-karate/soc-karate_rewired_all/soc-karate'
#outfile = 'soc-karate/soc-karate_all_exactHawkes_labeled.txt'
#alpha = 0.96 # theta / lambda_max -> value used in samira paper


## soc-dolphins
#src_graph = 'soc-dolphins/soc-dolphins.txt'
#dest_path = 'soc-dolphins/soc-dolphins_rewired_all/soc-dolphins'
#outfile = 'soc-dolphins/soc-dolphins_all_exactHawkes_labeled.txt'
#alpha = 0.95 # theta / lambda_max

## rt-retweet
#src_graph = 'rt-retweet/rt-retweet.txt'
#dest_path = 'rt-retweet/rt-retweet_rewired_all/rt-retweet'
#outfile = 'rt-retweet/rt-retweet_all_exactHawkes_labeled.txt'
#alpha = 0.98 # theta / lambda_max

## soc-anybeat
#src_graph = 'soc-anybeat/soc-anybeat.txt'
#dest_path = 'soc-anybeat/soc-anybeat_rewired_all/soc-anybeat'
#outfile = 'soc-anybeat/soc-anybeat_all_exactHawkes_labeled.txt'
#alpha = 0.10 # theta / lambda_max

## socfb-Reed98
#src_graph = 'socfb-Reed98/socfb-Reed98.txt'
#dest_path = 'socfb-Reed98/socfb-Reed98_rewired_all/socfb-Reed98'
#outfile = 'socfb-Reed98/socfb-Reed98_all_exactHawkes_labeled.txt'
#alpha = 0.99 # theta / lambda_max

## socfb-Caltech36
#src_graph = 'socfb-Caltech36/socfb-Caltech36.txt'
#dest_path = 'socfb-Caltech36/socfb-Caltech36_rewired_all/socfb-Caltech36'
#outfile = 'socfb-Caltech36/socfb-Caltech36_all_exactHawkes_labeled.txt'
#alpha = 0.99 # theta / lambda_max

## socfb-Haverford76
#src_graph = 'socfb-Haverford76/socfb-Haverford76.txt'
#dest_path = 'socfb-Haverford76/socfb-Haverford76_rewired_all/socfb-Haverford76'
#outfile = 'socfb-Haverford76/socfb-Haverford76_all_exactHawkes_labeled.txt'
#lpha = 0.99 # theta / lambda_max

## socfb-Simmons81
#src_graph = 'socfb-Simmons81/socfb-Simmons81.txt'
#dest_path = 'socfb-Simmons81/socfb-Simmons81_rewired_all/socfb-Simmons81'
#outfile = 'socfb-Simmons81/socfb-Simmons81_all_exactHawkes_labeled.txt'
#alpha = 0.99 # theta / lambda_max

## socfb-Simmons81
src_graph = 'socfb-Amherst41/socfb-Amherst41.txt'
dest_path = 'socfb-Amherst41/socfb-Amherst41_rewired_all/socfb-Amherst41'
outfile = 'socfb-Amherst41/socfb-Amherst41_all_exactHawkes_labeled.txt'
alpha = 0.99 # theta / lambda_max


###########################################
thread = 80  # number of threads used in e-clog
e_clog_mode = 'local' # select from local, unique, all

nswap = 1000      # num of swaps for rewiring
max_tries = 10000 # max_tries for double_edge_swap
num_graphs = 100 
max_gen = 100 # max_gen for exact_hawkes_maxgen

##############################################
print('reading edgelist ...\n')
G = nx.read_edgelist(src_graph)
print("getting theta ...\n")
start_time = time.time()
theta = alpha / lambda_max(G)
print("obtain theta after %s\n" % (time.time() - start_time))

#print("critical value is %f" % (1.0 / lambda_max(G)))

num_nodes = len(G.nodes)
num_edges = len(G.edges)

os.system('echo rewiring...')
#TE, node_list = gen_dpr_exact_all(G, nswap, max_tries, num_graphs, theta, path=dest_path)
#TE, node_list = gen_dpr_exact_maxgen_all(G, nswap, max_tries, num_graphs, theta, max_gen, path=dest_path)
TE, node_list = gen_dpr_exact_maxgen_all_theta(G, nswap, max_tries, num_graphs, alpha, max_gen, path=dest_path)
#TE, node_list = gen_dpr_exact_all_theta(G, nswap, max_tries, num_graphs, alpha, path=dest_path)
os.system('echo finish rewiring')

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


os.system('echo start e-clog')
os.system('./mul_eclog.sh %d %s %s %s' % (num_graphs-1, dest_path, thread, e_clog_mode))
os.system('echo finish e-clog')



###############################################
# convert edge-centric gfd to node-centric gfd
###############################################
X = []
for i in range(0, num_graphs):
    df = pd.read_csv(dest_path + '_graphlets_rewired%d.txt_%s_graphlet_freqeuncy_5_omp.txt' % (i, e_clog_mode), header=None, sep=' |: ')
    E = df.values
    for j in node_list[i]:
        X.append(list(sum_lgd(j-1, E)))

X = np.array(X)

#print(X.shape)
#print(TE.shape)
#print(node_list.shape)
df = pd.DataFrame(X)
df[46] = TE.flatten()
df[47] = node_list.flatten()
df.to_csv(outfile, header=False, index=False, sep=' ')

