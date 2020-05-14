import sys
import numpy as np
import pandas as pd
from simple_perceptron import *


def ej2():
    iterations = 100
    learning_rate = 0.1
    plot = False
    stop_early = True

    df = pd.read_excel("data/TP3-ej2-Conjunto_entrenamiento.xlsx", header=1)
    # remove all columns without name (empty cells from excel)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['bias'] = 1.0
    # reorder columns to have result y as last column
    df = df[[ 'bias','x1', 'x2', 'x3', 'y']]
    # reorder rows for ascending y
    df.sort_values('y', inplace=True)
    # slice df into training and test dfs
    #take every 5th row
    df_test = df.iloc[::5]
    # drop test examples from complete set to get training set
    df_train = df.drop(df_test.index)
    # transform dataframe back to list of lists to comply with subsequent method structures
    lol_train = df_train.values.tolist()
    lol_test = df_test.values.tolist()

    weights = [0.1,	0.30, 0.80, 0.1] # initial weights x1, x2, x3, bias

    weights = train_weights(lol_train, weights=weights, predict=relu, iterations=iterations, learning_rate=learning_rate, plot=plot, stop_early=stop_early)

    test_perceptron(lol_train, weights=weights, print_results=True)
    weights
if __name__ == '__main__':
    ej2()