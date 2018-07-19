import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error
from analysis_util import *
import networkx as nx
from graph_manip import *
import sys


def rank_by_guess(target, step_height, src = ['socfb-Reed98', 'socfb-Caltech36', 'socfb-Haverford76', 'socfb-Simmons81']):
    coef = []
    for i in src:
        df = pd.read_csv('%s/%s_linear_coef.txt' % (i, i), header=None)
        coef.append(df[0].values)

    coef = np.array(coef)
    coef = pd.DataFrame(coef.T, columns=src)
    coef = coef.iloc[1:,:]


    ave_coef = coef.mean(axis=1).values

    ## apply a step-function
    # only keep the coef whose absolute value >= step_height
    ave_coef_filtered = []
    for c in ave_coef:
        if np.abs(c) >= step_height:
            ave_coef_filtered.append(c)
        else:
            ave_coef_filtered.append(0.0)



    g = nx.read_edgelist('%s/%s.txt' % (target, target), nodetype=int)
    df = pd.read_csv('%s/eclog/%s_graphlets.txt_local_graphlet_freqeuncy_5_omp.txt' % (target, target), header=None, sep=' |: ')

    node_list = list(g.nodes)

    ###############################################
    # convert edge-centric gfd to node-centric gfd
    ###############################################
    X = []
 
    E = df.values
    for j in node_list:
        X.append(list(sum_lgd(j-1, E)))

    X = np.array(X)

    df = pd.DataFrame(X)


    df = df.apply(log_freq_count, axis=1)


    ec = []
    for i in range(0, df.shape[0]):
        ec.append(np.dot(df.iloc[i,:], ave_coef_filtered))

    df1 = pd.DataFrame(list(g.nodes), columns=['node'])
    df1['coeff'] = ec
    df2 = df1.sort_values(by='coeff', ascending=False)

    df2.to_csv('%s/rank_node_coeff.txt' % (target), header=False, index=False, sep=' ')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: not enough args")
        print("usage: python3 rank_node.py network_name step_height")
        print("ex: python3 rank_node.py socfb-Swarthmore42 0.2")
        exit()
    
    target = sys.argv[1]
    step_height = np.float(sys.argv[2])
    
    rank_by_guess(target, step_height)


