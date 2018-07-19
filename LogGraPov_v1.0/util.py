# utilities
import numpy as np
import pandas as pd
import json

# parse config (json format) for rewiring
def parse_config_rewire(file_name):
	with open(file_name) as file:
		config = json.load(file)
		
	src_graph = config['src_graph']
	nswap = config['nswap']
	max_tries = config['max_tries']
	num_graphs = config['num_graphs']
	dest_path = config['dest_path']

	return src_graph, nswap, max_tries, num_graphs, dest_path

# parse config (json format) for sample hawkes
def parse_config_sample_hawkes(file_name):
	with open(file_name) as file:
		config = json.load(file)
	
	src_graph = config['src_graph']
	max_lamb = config['max_lamb']
	alpha = config['alpha']
	num_samples = config['num_samples']
	distinct = config['distinct']
	output = config['output']
	
	return src_graph, max_lamb, alpha, num_samples, distinct, output

# parse config (json format) for sample hawkes
def parse_config_exact_hawkes(file_name):
	with open(file_name) as file:
		config = json.load(file)
	
	src_graph = config['src_graph']
	max_lamb = config['max_lamb']
	maxgen = config['maxgen']
	alpha = config['alpha']
	output = config['output']
	
	return src_graph, max_lamb, maxgen, alpha, output


# parse config (json format) for sample hawkes
def parse_config_eclog(file_name):
	with open(file_name) as file:
		config = json.load(file)

	src_graph = config['src_graph']
	num_graphs = config['num_graphs']
	num_threads = config['num_threads']
	eclog_mode = config['eclog_mode']

	return src_graph, num_graphs, num_threads, eclog_mode

# parse config (json format) for edge2node_eclog
def parse_config_edge2node(file_name):
	with open(file_name) as file:
		config = json.load(file)

	src_path = config['src_path']
	num_graphs = config['num_graphs']
	eclog_mode = config['eclog_mode']

	return src_path, num_graphs, eclog_mode

# parse config (json format) for linear regression
def parse_config_linear_regression(file_name):
	with open(file_name) as file:
		config = json.load(file)

	src_x = config['src_x']
	src_y = config['src_y']
	num_graphs = config['num_graphs']
	test_size = config['test_size']
	lfc = config['lfc']
	ylog = config['ylog']
	parafile = config['parafile']

	return src_x, src_y, num_graphs, test_size, lfc, ylog, parafile


# parse config (json format) for all
def parse_config(file_name):
	with open(file_name) as file:
		config = json.load(file)

	return config

# split the graphs into training set and test set
# src_x are the files that contain data, e.x. rewired/socfb-Reed98
# src_y are the hawkes count, ex. hawkes/sample_hawkes
# if lfc is true, convert the graphlet count to log frequency distribution
# if ylog is true, convert the hawkes event count to log count
def train_test_file_split(src_x, src_y, num_graphs, test_size=0.3, lfc=True, ylog=True):
    num_test = num_graphs * test_size
    num_train = num_graphs - num_test

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(0, num_graphs):
        df1 = pd.read_csv(src_x + "_node_centric_graphlet_frequency%d.txt" % (i), header=None, sep=' ')
        df2 = pd.read_csv(src_y + "%d.txt" % (i), header=None, sep=' ')

        if lfc:
            df1 = df1.apply(log_freq_count, axis=1)

        if ylog:
            y = np.log(df2[1].values)
        else:
            y = df2[1].values

        if i <= num_train:
            x_train += df1.values.tolist()
            y_train += y.tolist()
        else:
            x_test += df1.values.tolist()
            y_test += y.tolist()

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)



# turn counts into frequencies
def freq_count(d):
    l = (d+1) / np.sum(d+1)
    #print(np.sum(d))
    return l

# turn counts into log(frequency)
def log_freq_count(d):
    return np.log(freq_count(d))


# identify the k best candidates
def find_k_best(k, model, df): # df -- local graphlet distribution, hawkes count, node
	X = df.iloc[:, 0:46]
	y = df[46]
	
	y_predict = model.predict(X)

	df['true'] = y
	df['predict'] = y_predict

	knode_predict = df.sort_values(by='predict', axis=0, ascending=False).iloc[0:k, 47].values
	score_predict = np.exp(df.sort_values(by='predict', axis=0, ascending=False).iloc[0:k, :]['predict'].values)
	knode_true = df.sort_values(by='true', axis=0, ascending=False).iloc[0:k, 47].values
	score_true = df.sort_values(by='true', axis=0, ascending=False).iloc[0:k, :]['true'].values

	return knode_predict, knode_true, score_predict, score_true


