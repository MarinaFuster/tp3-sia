import numpy as np
import pandas as pd
import random
import sys
from non_linear_perceptron import train_weights_nonlinear
from sklearn import preprocessing

RANDOM = 0
SORTED = 1

def ej2():
    epochs = 100
    learning_rate = 0.1
    plot = False
    stop_early = True

   
    matrix = get_matrix_from_xlsx("data/TP3-ej2-Conjunto_entrenamiento.xlsx")
    matrix = include_bias_feature(normalize_data(matrix))

    weights = [0.1,	1., 1., 1.] # initial weights w0 (bias), w1, w2, w3

    training_data, samples_to_predcit = select_data(matrix, SORTED)

    weights = train_weights_nonlinear(matrix, weights, learning_rate=learning_rate, stop_early=stop_early)
    print(weights)

# appends column of ones for bias
def include_bias_feature(matrix):
    return np.column_stack((np.ones((len(matrix),1)), normalize_data(matrix)))

# normalizes data
def normalize_data(matrix):
	# axis used to normalize the data along. If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
	return preprocessing.normalize(matrix, axis=0)

# Returns training data and testing data
def select_data(matrix, process):
    training_data = []
    samples_to_predict = []
    if process == RANDOM:
        indexes = random.sample(range(0, len(matrix)), len(matrix))
        for i in range(len(matrix)):
            if i % 2 == 0:
                training_data.append(matrix[indexes[i]])
            else:
                samples_to_predict.append(matrix[indexes[i]])
    if process == SORTED:
        matrix = matrix[matrix[:,-1].argsort()]
        for i in range(len(matrix)):
            if i % 2 == 0:
                training_data.append(matrix[i])
            else:
                samples_to_predict.append(matrix[i])
    return np.array(training_data), np.array(samples_to_predict)
        

def get_matrix_from_xlsx(file):
    df = pd.read_excel(file, header=1)
    # remove all columns without name (empty cells from excel)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df['bias'] = 1.0
    # reorder columns to have result y as last column
    df = df[['x1', 'x2', 'x3', 'y']]
    # reorder rows for ascending y
    df.sort_values('y', inplace=True)
    # slice df into training and test dfs
    #take every 5th row
    df_test = df.iloc[::5]
    # drop test examples from complete set to get training set
    df_train = df.drop(df_test.index)
    # transform dataframe back to list of lists to comply with subsequent method structures
    lol_train = df_train.values.tolist()
    return lol_train

if __name__ == '__main__':
    ej2()