import numpy as np
import networkx as nx
import scipy.stats as stats
from point_process import *
from graph_manip import *

target = 'socfb-Swarthmore42'
alpha = 0.99
G = nx.read_edgelist('%s/%s.txt' % (target, target), nodetype=int)

theta = alpha / lambda_max(G)
        
T = exact_hawkes_all(G, theta)
nodes = list(G.nodes)

df = pd.DataFrame(nodes, columns=['node'])
df['hawkes'] = T

df = df.sort_values(by='hawkes', ascending=False)
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
df['rank'] = df.index
df = df.sort_values(by='node')
df.to_csv('%s/node_rank_hawkes.txt' % (target), header=False, index=False, sep=' ')


