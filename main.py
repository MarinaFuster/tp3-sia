from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from simple_perceptron import train_weights, step_function
from multilayer_perceptron import Multilayer_perceptron, import_numdata, print_size
from ej2 import select_data, get_matrix_from_xlsx, normalize_data

def main():
	epochs		    = 100
	learning_rate  	= 0.5
	plot_iteration	= True
	stop_early 		= True

					# x1 	 x2       y
	and_matrix = [	[1.00,	1.00,	1.0],
					[-1.00,	1.00,	-1.00],
					[1.00,	-1.00,	-1.00],
					[-1.00,	-1.00,	-1.00]]
	
					# x1 	 x2       y
	xor_matrix = [	[1.00,	1.00,	-1.0],
					[-1.00,	1.00,	1.00],
					[1.00,	-1.00,	1.00],
					[-1.00,	-1.00,	-1.00]]

	weights= [	 0.1,	0.30,  0.80		] # initial weights

	train_weights(and_matrix, weights=weights, predict=step_function, epochs=epochs, learning_rate=learning_rate, plot=True, stop_early=stop_early)

def multi_layer_xor():
	# Multi layer perceptron
	xor_matrix_multi = [	[1.00,	1.00],
							[-1.00,	1.00],
							[1.00,	-1.00],
							[-1.00,	-1.00]]

	xor_matrix_multi_expected = [[1],[0],[0],[1]]
	mlp = Multilayer_perceptron([5 ,1],len(xor_matrix_multi[0]))

	np_xor = np.array(xor_matrix_multi)
	np_xor_expected = np.array(xor_matrix_multi_expected)




	mlp.train_weights(np_xor, np_xor_expected)
	print(mlp.predict(np_xor))

def multi_layer_primos():
	data = 	import_numdata()
	# zero_one = np.array(data[0:2,:])
	# zero_one_expected = np.array([0,1])
	data_expected = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
	mlp = Multilayer_perceptron([10, 20 ,1],len(data[0]))
	

	merged_data = np.column_stack((data, data_expected))

	training_data_merged, test_data_merged = select_data(merged_data, 0)

	training_data = training_data_merged[:,:-1]
	training_data_expected = training_data_merged[:,-1]
	# El vector de training data expected tiene que ser una matriz
	training_data_expected = training_data_expected.reshape(len(training_data_expected), 1)
	test_data = test_data_merged[:,:-1]
	test_data_expected = test_data_merged[:,-1]
	test_data_expected = test_data_expected.reshape(len(test_data_expected), 1)



	mlp.train_weights(training_data, training_data_expected)
	prediction = mlp.predict(training_data)

	print_size(prediction, "prediction")
	print_size(test_data_expected, "expected")
	

	error = training_data_expected - prediction
	print(error)
	print((error**2).mean())

def multi_layer_ej2():
	training_data_merged  = np.array(normalize_data(get_matrix_from_xlsx("data/TP3-ej2-Conjunto_entrenamiento.xlsx")))
	training_data = training_data_merged[:,:-1]
	training_data_expected = training_data_merged[:,-1]
	training_data_expected = training_data_expected.reshape(len(training_data_expected), 1)
	mlp = Multilayer_perceptron([50 ,1],len(training_data[0]))
	mlp.train_weights(training_data, training_data_expected)


	# print(training_data)
	# print(training_data_expected)

	# prediction = mlp.predict(training_data).T
	# error = training_data_expected - prediction
	# print(error)
	# print(np.sum(np.sqrt(error **2 )) / len(training_data))
if __name__ == '__main__':
	# main()
	multi_layer_primos()
	# multi_layer_ej2()
	# multi_layer_xor()
