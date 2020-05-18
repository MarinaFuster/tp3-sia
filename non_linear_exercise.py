import numpy as np
import pandas as pd
import random
import sys
from non_linear_perceptron import train_weights_nonlinear, test_perceptron, non_linear_perceptron_info
from sklearn import preprocessing

RANDOM = 0
SORTED = 1

def run_non_linear_for_exercise_two(config):
    epochs = config["non_linear_perceptron"]["epochs"]
    learning_rate = config["non_linear_perceptron"]["learning_rate"]
    beta = config["non_linear_perceptron"]["beta"]
    selection_method = config["non_linear_perceptron"]["selection_method"]

    matrix = get_matrix_from_xlsx("data/TP3-ej2-Conjunto_entrenamiento.xlsx")
    matrix = np.array(matrix)
    outputs = matrix[:, -1]
    max_value = np.max(outputs)
    min_value = np.min(outputs)
    
    matrix = include_bias_feature(matrix)
    matrix = normalize_data(matrix, min_value, max_value)
    
    weights = [0.1, 1.0, 1.0, 1.0]
    print(weights)

    training_data, samples_to_predict = select_data(matrix, selection_method)

    print(non_linear_perceptron_info(learning_rate, beta, epochs, selection_method))
    weights = train_weights_nonlinear(matrix, weights, learning_rate=learning_rate, epochs=epochs, stop_early=True)
    predictions, groundtruths, RSEs = test_perceptron(matrix, weights, print_results=True)

def random_weights(matrix):
    if len(matrix) == 0:
        return []
    return np.random.rand(4)

# appends column of ones for bias
def include_bias_feature(matrix):
    return np.column_stack((np.ones((len(matrix),1)), matrix))

# normalizes data
def normalize_data(matrix, min_value, max_value):
    for i in range(0, len(matrix)):
        matrix[i][-1] =  (matrix[i][-1] - min_value) / (max_value - min_value)
    return matrix

# def denormalize_data(matrix, min_value, max_value):
#     for i in range(0, len(matrix)):
#         matrix[i] = (matrix[i] * (max_value - min_value)) + min_value
#     return matrix

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
    #df_test = df.iloc[::5]
    # drop test examples from complete set to get training set
    #df_train = df.drop(df_test.index)
    # transform dataframe back to list of lists to comply with subsequent method structures
    #lol_train = df_train.values.tolist()
    lol_data = df.values.tolist()
    return lol_data
