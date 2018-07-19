import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from analysis_util import *
import networkx as nx

k = 7 # k best candidates

#src_graph = 'soc-karate/soc-karate.txt'
#src_graph = 'soc-dolphins/soc-dolphins.txt'
#src_graph = 'rt-retweet/rt-retweet.txt'
#src_graph = 'soc-firm-hi-tech/soc-firm-hi-tech.txt'
#src_graph = 'socfb-Reed98/socfb-Reed98.txt'
#src_graph = 'socfb-Caltech36/socfb-Caltech36.txt'
#src_graph = 'socfb-Simmons81/socfb-Simmons81.txt'
src_graph = 'soc-wiki-vote/soc-wiki-vote.txt'
#src_graph = 'rt-twitter-copen/rt-twitter-copen.txt'
#src_graph = 'socfb-Haverford76/socfb-Haverford76.txt'

#infile = 'soc-karate/soc-karate_all_exactHawkes_labeled.txt'
#infile = 'soc-dolphins/soc-dolphins_all_exactHawkes_labeled.txt'
infile = 'soc-wiki-vote/soc-wiki-vote_all_exactHawkes_labeled.txt'
#infile = 'soc-firm-hi-tech/soc-firm-hi-tech_all_exactHawkes_labeled.txt'
#infile = 'rt-retweet/rt-retweet_all_exactHawkes_labeled.txt'
#infile = 'rt-twitter-copen/rt-twitter-copen_all_exactHawkes_labeled.txt'
#infile = 'socfb-Reed98/socfb-Reed98_all_exactHawkes_labeled.txt'
#infile = 'socfb-Caltech36/socfb-Caltech36_all_exactHawkes_labeled.txt'
#infile = 'socfb-Simmons81/socfb-Simmons81_all_exactHawkes_labeled.txt'
#infile = 'socfb-Haverford76/socfb-Haverford76_all_exactHawkes_labeled.txt'

#parafile = 'socfb-Reed98/socfb-Reed98_linear_coef.txt'
#parafile = 'socfb-Caltech36/socfb-Caltech36_linear_coef.txt'
#parafile = 'socfb-Simmons81/socfb-Simmons81_linear_coef.txt'
parafile = 'soc-wiki-vote/soc-wiki-vote_linear_coef.txt'
#parafile = 'rt-twitter-copen/rt-twitter-copen_linear_coef.txt'
#parafile = 'socfb-Haverford76/socfb-Haverford76_linear_coef.txt'


df = pd.read_csv(infile, header=None, sep=' ')

X = df.iloc[:,0:46].values
y = df[46].values

l = [e for e in X.flatten() if e == 0.]

num_zeros = len(l) / len(X.flatten())
print("num of zeros in X: %.3f\n" % num_zeros)

df_x = pd.DataFrame(X)
df_x = df_x.apply(log_freq_count, axis=1)
X = df_x.values
y = np.log(y)


#sns.distplot(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=123)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# write coef 
with open(parafile, 'w') as file:
	file.write('%s\n' % (lr.intercept_))
	for s in lr.coef_:
		file.write('%s\n' % (s))

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("r2 = %f" % r2)
print("mse = %f" % mse)
#plt.scatter(list(range(0, len(y))), y)
#plt.scatter(list(range(0, len(X_test))), y_pred, label='pred')
#plt.scatter(list(range(0, len(X_test))), y_test, label='true')
#plt.legend(loc=0)

#plt.scatter(y_test, y_pred, alpha=0.5)

#plt.xticks(np.arange(0,3))
#plt.yticks(np.arange(0,3))
#plt.xlabel('exact hawkes event counts')
#plt.ylabel('predict event counts')
#plt.plot(np.arange(0, 3,0.1), np.arange(0,3,0.1), 'r')
#plt.show()



"""
G = nx.read_edgelist(src_graph)
n = len(list(G.nodes))


df1 = df.iloc[0:n, :]
knode_predict, knode_true, score_predict, score_true = find_k_best(k, lr, df1)
print('\n')
print('----- Predict -----')
print(knode_predict)
print(score_predict)
print('----- Hawkes ------')
print(knode_true)
print(score_true)


val_map1 = dict(zip(knode_predict, np.repeat(1, k)))
val_map2 = dict(zip(knode_true, np.repeat(1, k)))
values1 = [val_map1.get(np.int(node), 0.25) for node in G.nodes()]
values2 = [val_map2.get(np.int(node), 0.25) for node in G.nodes()]

p = []
for i in range(0, n):
    p.append(tuple(np.random.rand(2)))

pos = dict(zip(G.nodes, p))

plt.figure(1)
plt.title('Predict')
nx.draw(G, pos=pos, cmap=plt.get_cmap('rainbow'), node_color=values1, alpha=0.5, with_labels=True)
plt.figure(2)
plt.title('Hawkes')
nx.draw(G, pos=pos, cmap=plt.get_cmap('rainbow'), node_color=values2, alpha=0.5, with_labels=True)
plt.show()
"""
