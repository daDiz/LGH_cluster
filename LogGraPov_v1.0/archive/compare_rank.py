import numpy as np
import pandas as pd
import sys


def compare(target, first):
    df_pred = pd.read_csv('%s/rank_node_coeff.txt' % (target), header=None, sep=' ')
    df_true = pd.read_csv('%s/rank_node_hawkes.txt' % (target), header=None, sep=' ')

    correct_in = 0
    correct_rank = 0
    for i in range(0, first):
        if df_pred.iloc[i, 0] in df_true.iloc[0:first, 0].values:
            correct_in += 1
            if df_pred.iloc[i, 0] == df_true.iloc[i, 0]:
                correct_rank += 1
    
    print('first %d out of %d nodes:' % (first, df_pred.shape[0]))
    print('%f are correct' % (correct_in / first))
    print('%f are ranked correctly' % (correct_rank / first))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: not enough args")
        print("usage: python3 compare_rank.py socfb-Swarthmore42 100")
        exit()

    target = sys.argv[1]
    first = np.int(sys.argv[2])
    compare(target, first)


   
