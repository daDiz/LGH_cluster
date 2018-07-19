# functions for point process
import time
import networkx as nx
import numpy as np
import pandas as pd
from graph_manip import *

def is_converge(G, theta):
    A = nx.adj_matrix(G).todense()
    e, V = np.linalg.eig(A)
    if min(e) * theta <= -1 or max(e) * theta >= 1:
        return False
    else:
        return True

def ave_lgd(i, E):
    E1 = E[(E[:,0] == i) | (E[:,1] == i), 2:]
    return np.mean(E1, axis=0)

def sum_lgd(i, E):
    E1 = E[(E[:,0] == i) | (E[:,1] == i), 2:]
    return np.sum(E1, axis=0)

def propagate(i, theta, E, T):
    l = E[E[:,0]==i, 1]
    N = len(l)
    if N>0:
        v = (np.random.rand(N) < theta)
        l = l[v]
    T = T + len(l)
    return l, T

def propagate_distinct(i, theta, E, T, visited):

    # neighbors of i
    l = E[E[:,0]==i, 1] 
    N = len(l)
    if N > 0:
        v = (np.random.rand(N) < theta)
        l = l[v]
        l = np.setdiff1d(l, visited, assume_unique=True)
        visited = np.append(visited, l)
        
    T = T + len(l)

    return l, T, visited

# do sample hawkes from node i
# if distinct is true, each node is visited at most once
def sample_hawkes_i(i, E, num_samples, theta, distinct=True):
    T = np.ones(num_samples)
    
    for k in range(0, num_samples):
        node_list = np.array([i])
        visited = np.array([i])
        
        while len(node_list) > 0:
            if distinct:
                l, T[k], visited = propagate_distinct(node_list[0], theta, E, T[k], visited)
            else:
                l, T[k] = propagate(node_list[0], theta , E , T[k])
            
            node_list = node_list[1:]
            node_list = np.append(node_list,l)
    
    return np.mean(T)


# do sample hawkes from all the nodes in a graph
# if distinct is true, each node is visited at most once
def sample_hawkes_all(G, num_samples, theta, distinct=True):
    E = to_sym_edgelist(G)

    nodes = G.nodes
    T = []
    for i in nodes:
        T.append(sample_hawkes_i(i, E, num_samples, theta, distinct))

    return np.array(T)


def sample_hawkes_i_old(i, E, Nsamples, theta):
    T=np.ones(Nsamples)
    
    for k in range(0, Nsamples):
        node_list = np.array([i])
        while len(node_list) > 0:
            #print(node_list)
            l, T[k] = propagate(node_list[0], theta , E , T[k])
            
            node_list = node_list[1:]

            if len(l) != 0:
                node_list=np.append(node_list,l)

    
    return np.mean(T)


def exact_hawkes_i(i, G, theta): # i is the node num
    A = nx.adj_matrix(G).todense()
    e, V = np.linalg.eig(A)
    
    k = list(G.nodes).index(i) # index of node i in G.nodes
    #print(k)
    D = np.diag(e) * theta
    D = 1.0 / (1.0-D) - 1.0 # q/(1-q)
    M = np.dot(np.dot(V, D), np.linalg.inv(V))
    N = A.shape[0] # num of nodes
    init = np.zeros(N)
    init[k] = 1
    init = init.reshape((N,1))
    
    return np.sum(np.matmul(M, init).real)

def exact_hawkes_all(G, theta): # hawkes is performed node by node in the order of G.nodes
    A = nx.adj_matrix(G).todense()
    e, V = np.linalg.eig(A)
    D = np.diag(e) * theta
    D = 1.0 / (1.0-D) - 1.0 # q/(1-q)
    M = np.dot(np.dot(V, D), np.linalg.inv(V))
    N = A.shape[0] # num of nodes
    ones = np.ones(N).reshape((N,1))
    
    T = np.matmul(M, ones)

    return np.array(T.real.flatten().tolist()[0])


def exact_hawkes(G, theta): # i is a node
    A = nx.adj_matrix(G).todense()
    e, V = np.linalg.eig(A)
    
    D = np.diag(e) * theta
    D = 1.0 / (1.0-D) - 1.0 # q/(1-q)
    M = np.dot(np.dot(V, D), np.linalg.inv(V))
    N = A.shape[0] # num of nodes
    
    ones = np.ones(N)
    res = np.matmul(M, ones.reshape((N, 1)))
    TE = np.matmul(ones.reshape((1,N)), res).real[0,0] / N
    return TE


def exact_hawkes_maxgen_all(G, maxgen, theta):
    #print("calculating adj matrix\n")
    #start_time =  time.time()
    A = nx.adj_matrix(G).todense() * theta
    #print("finish adj matrix after %s\n" % (time.time() - start_time))
    N = A.shape[0]
    M = np.zeros((N, N))
    A_tmp = np.diag(np.ones(N))
    #print("calculating exact hawkes\n")
    #start_time = time.time()
    for i in range(0, maxgen):
        A_tmp = np.matmul(A_tmp, A)
        M += A_tmp
        
    #print("matrix sum done after %s\n" % (time.time() - start_time))
    ones = np.ones(N).reshape((N,1))
    T = np.matmul(M, ones)

    return T.real.flatten().tolist()


def exact_hawkes_maxgen_all_old(G, maxgen, theta):
    A = nx.adj_matrix(G).todense() * theta
    N = A.shape[0]
    M = np.zeros((N, N))
    for i in range(0, maxgen):
        M += np.linalg.matrix_power(A, i+1)

    ones = np.ones(N).reshape((N,1))
    T = np.matmul(M, ones)

    return T.real.flatten().tolist()



def exact_hawkes_maxgen(G, maxgen, theta):
    A = nx.adj_matrix(G).todense() * theta
    N = A.shape[0]
    M = np.zeros((N, N))
    for i in range(0, maxgen):
        M += np.linalg.matrix_power(A, i+1)

    ones = np.ones(N).reshape((N,1))
    TE = np.sum(np.matmul(M, ones) / N)

    return TE



