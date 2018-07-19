#!/usr/bin/env python3

import sys
sys.path.insert(0, './')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from util import *
import networkx as nx


def main():
    if len(sys.argv) != 2:
        print("Error: wrong number of args")
        print("usage: python3 build_linear_model.py config_file")
        print("ex: python3 build_linear_model.py config.json")
        exit()

    config_file = sys.argv[1]
    config = parse_config(config_file)

    src_x = config['linear_regression']['src_x']
    src_y = config['linear_regression']['src_y']
    num_graphs = config['linear_regression']['num_graphs']
    test_size = config['linear_regression']['test_size']
    lfc = config['linear_regression']['lfc']
    ylog = config['linear_regression']['ylog']
    parafile = config['linear_regression']['parafile']
    
    
    X_train, y_train, X_test, y_test = train_test_file_split(src_x, src_y, num_graphs, test_size=test_size, lfc=lfc, ylog=ylog)
    

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

    return


if __name__ == "__main__":
    main()
