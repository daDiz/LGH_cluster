#######################
# manipulate graphs
#######################
import numpy as np
import networkx as nx
import scipy.stats as stats
from point_process import *
import scipy.sparse as sparse

# second order average degree of a graph
def mean_degree_2nd(G):
    w2 = 0.0
    w1 = 0.0
    for n, d in G.degree():
        w2 += d ** 2
        w1 += d

    return w2 / w1

# find largest degree of a graph
def degree_max(G):
    d = dict(G.degree)
    return sorted(d.items(), key=lambda kv: kv[1])[-1][1]


def degree_max2(G):
    n = []
    d = []
    for i, j in G.degree:
       n.append(i)
       d.append(j)
    return n[np.argmax(d)]


# find nodes w/ maximum degree
def max_degree_nodes(G):
    dm = degree_max(G)
    nodes = []
    for v, d in G.degree:
        if d == dm:
            nodes.append(v)
    return np.array(nodes)

# find lambda max for a graph
def lambda_max(G):
    A = nx.adj_matrix(G).asfptype()
    w = sparse.linalg.eigsh(A, k=1, which='LA', return_eigenvectors=False)
    return w[0]

def lambda_max_old(G):
    A = nx.adj_matrix(G)
    w, v = np.linalg.eig(A.todense())
    return np.max(w).real

# add constant weight to edges
def add_weight_const(G, w=0.5): # add weights inplace
    for i, j in G.edges:
        G[i][j]['weight'] = w
    
    return

# add random weights to edges generated from a truncated normal distribution
def add_weight_rand(G, mu=0.5, sigma=0.2): # add weights inplace
    X = stats.truncnorm((0-mu)/sigma,(1-mu)/sigma,loc=mu,scale=sigma)
    w = X.rvs(len(G.edges))
    k = 0
    for i, j in G.edges:
        G[i][j]['weight'] = w[k]
    
    return

# convert a graph to a symmetric edge list
def to_sym_edgelist(g):
    E = []
    for e in g.edges:
        E.append([e[0], e[1]])
        E.append([e[1], e[0]])
    return np.array(E)

# generate input for e-clog
def gen_eclog_input(src_path, num_graphs):
    for i in range(0, num_graphs):
        G = nx.read_edgelist(src_path + '_rewired%d.txt' % (i), nodetype=int)
        
        num_nodes = len(list(G.nodes))
        num_edges = len(list(G.edges))

        with open(src_path + '_graphlets_rewired%d.txt' % (i), 'w') as file:
            file.write('%d %d\n' % (num_nodes, num_edges))
            for e1, e2 in G.edges:
                x = e1 - 1
                y = e2 - 1
                file.write('%s %s\n' % (x,y))

    return

# convert edge-centric graphlet counts to node-centric graphlet counts
def edge2node(src_path, num_graphs, eclog_mode):
    for i in range(0, num_graphs):
        g = nx.read_edgelist(src_path + '_rewired%d.txt' % (i), nodetype=int)
        node_list = g.nodes
        X = []
        df = pd.read_csv(src_path + '_graphlets_rewired%d.txt_%s_graphlet_freqeuncy_5_omp.txt' % (i, eclog_mode), header=None, sep=' |: ')
        E = df.values
        for j in node_list:
            X.append(list(sum_lgd(j-1, E)))

        df1 = pd.DataFrame(X)
        df1.to_csv(src_path + '_node_centric_graphlet_frequency%d.txt' % (i), header=False, index=False, sep=' ')
    
    return


# degree preserve rewiring
def dpr_rewire(G, nswap, max_tries, num_graphs, path):
    new_G = G.copy()
    
    i = 0
    while i < num_graphs:
        nx.write_edgelist(new_G, data=False, path=path+'_rewired%d.txt' % (i))
        i += 1
        nx.double_edge_swap(new_G, nswap=nswap, max_tries=max_tries)
    
    return

def gen_dpr_exact_all(G, nswap, max_tries, num_graphs, theta, path):
    new_G = G.copy()
    T = [] # a list of event counts
    node_list = [] # a list of init nodes
    i = 0
    while i < num_graphs:
        if is_converge(new_G, theta):
            nx.write_edgelist(new_G, data=False, path=path+'_rewired%d.txt' % (i))
            T_i = exact_hawkes_all(new_G, theta)
            nodes = [np.int(e) for e in list(new_G.nodes)]
            i += 1
            T.append(T_i)
            node_list.append(nodes)
        nx.double_edge_swap(new_G, nswap=nswap, max_tries=max_tries)
    
    return np.array(T), np.array(node_list)



def gen_dpr_exact_maxgen_all(G, nswap, max_tries, num_graphs, theta, max_gen, path):
    new_G = G.copy()
    T = [] # a list of event counts
    node_list = [] # a list of init nodes
    i = 0
    while i < num_graphs:
        nx.write_edgelist(new_G, data=False, path=path+'_rewired%d.txt' % (i))
        T_i = exact_hawkes_maxgen_all(new_G, max_gen, theta)
        nodes = [np.int(e) for e in list(new_G.nodes)]
        i += 1
        T.append(T_i)
        node_list.append(nodes)
        nx.double_edge_swap(new_G, nswap=nswap, max_tries=max_tries)
    
    return np.array(T), np.array(node_list)

# perform exact hawkes using graph-specific theta (i.e. different rewirings different theta's)
# theta = alpha / lambda_max
def gen_dpr_exact_maxgen_all_theta(G, nswap, max_tries, num_graphs, alpha, max_gen, path):
    new_G = G.copy()
    T = [] # a list of event counts
    node_list = [] # a list of init nodes
    i = 0
    while i < num_graphs:
        nx.write_edgelist(new_G, data=False, path=path+'_rewired%d.txt' % (i))
        
        theta = alpha / lambda_max(new_G)

        T_i = exact_hawkes_maxgen_all(new_G, max_gen, theta)
        nodes = [np.int(e) for e in list(new_G.nodes)]
        i += 1
        T.append(T_i)
        node_list.append(nodes)
        nx.double_edge_swap(new_G, nswap=nswap, max_tries=max_tries)
    
    return np.array(T), np.array(node_list)


# perform exact hawkes using graphe-specific theta
def gen_dpr_exact_all_theta(G, nswap, max_tries, num_graphs, alpha, path):
    new_G = G.copy()
    T = [] # a list of event counts
    node_list = [] # a list of init nodes
    i = 0
    while i < num_graphs:
        nx.write_edgelist(new_G, data=False, path=path+'_rewired%d.txt' % (i))

        theta = alpha / lambda_max(new_G)
        
        T_i = exact_hawkes_all(new_G, theta)
        nodes = [np.int(e) for e in list(new_G.nodes)]
        i += 1
        T.append(T_i)
        node_list.append(nodes)
        nx.double_edge_swap(new_G, nswap=nswap, max_tries=max_tries)
    
    return np.array(T), np.array(node_list)





