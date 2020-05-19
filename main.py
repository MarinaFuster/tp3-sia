from __future__ import print_function
import json
import math
import numpy as np
import os
import pandas as pd
import sys
from graphics_plot import plot_multi_layer_perceptron
from multilayer_perceptron import Multilayer_perceptron, import_numdata, print_size
from simple_perceptron import train_weights, step_function, simple_perceptron_info
from non_linear_exercise import select_data, get_matrix_from_xlsx, include_bias_feature, normalize_data, run_non_linear_for_exercise_two

AND = 0
XOR = 1

# Load and validate configuration
current_path = os.path.dirname(os.path.realpath(__file__))

def simple_perceptron_ej1(problem, config):
	if problem == AND:
		epochs		    = config["simple_perceptron_and"]["epochs"]
		learning_rate  	= config["simple_perceptron_and"]["learning_rate"]
		plot	= config["simple_perceptron_and"]["plot"]
		stop_early 		= config["simple_perceptron_and"]["stop_early"]
	else:
		epochs		    = config["simple_perceptron_xor"]["epochs"]
		learning_rate  	= config["simple_perceptron_xor"]["learning_rate"]
		plot	= config["simple_perceptron_xor"]["plot"]
		stop_early 		= config["simple_perceptron_xor"]["stop_early"]

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

	matrix = []
	if problem == AND: matrix = and_matrix
	if problem == XOR: matrix = xor_matrix

	print(simple_perceptron_info(learning_rate, epochs, problem))
	train_weights(matrix, weights=weights, predict=step_function, epochs=epochs, learning_rate=learning_rate, plot=plot, stop_early=stop_early)

def multi_layer_xor(config):
	# Multi layer perceptron
	xor_matrix_multi = [	[1.00,	1.00],
							[-1.00,	1.00],
							[1.00,	-1.00],
							[-1.00,	-1.00]]

	xor_matrix_multi_expected = [[1],[0],[0],[1]]
	mlp = Multilayer_perceptron(config["multi_layer_perceptron_xor"]["layers"],len(xor_matrix_multi[0]))

	if config["multi_layer_perceptron_xor"]["plot"]:
		plot_multi_layer_perceptron(config["multi_layer_perceptron_xor"]["layers"], len(xor_matrix_multi_expected))

	np_xor = np.array(xor_matrix_multi)
	np_xor_expected = np.array(xor_matrix_multi_expected)

	mlp.train_weights(np_xor, np_xor_expected)
	print(mlp.predict(np_xor))

def multi_layer_primos(config):
	data = 	import_numdata()
	# zero_one = np.array(data[0:2,:])
	# zero_one_expected = np.array([0,1])

	data_expected = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])

	mlp = Multilayer_perceptron(config["multi_layer_perceptron_primes"]["layers"],len(data[0]))	
	merged_data = np.column_stack((data, data_expected))

	if config["multi_layer_perceptron_primes"]["plot"]:
		plot_multi_layer_perceptron(config["multi_layer_perceptron_primes"]["layers"], len(data_expected))
	
	training_data_merged, test_data_merged = select_data(merged_data, 0, math.ceil(len(merged_data) / 2))

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
	print("Errors: {}".format(error))
	print("Mean Error: {}".format((error**2).mean()))

def multi_layer_ej2(config):
	
	matrix = np.array(get_matrix_from_xlsx("data/TP3-ej2-Conjunto_entrenamiento.xlsx"))
	outputs = matrix[:, -1]
	min_value = np.min(matrix)
	max_value = np.max(matrix)

	matrix = include_bias_feature(matrix)
	matrix = normalize_data(matrix, min_value, max_value)
	training_data_merged  = np.array(matrix)
	training_data = training_data_merged[:,:-1]
	training_data_expected = training_data_merged[:,-1]
	training_data_expected = training_data_expected.reshape(len(training_data_expected), 1)
	mlp = Multilayer_perceptron(config["multi_layer_perceptron_dataset_ex_2"]["layers"],len(training_data[0]))

	if config["multi_layer_perceptron_dataset_ex_2"]["plot"]:
		plot_multi_layer_perceptron(config["multi_layer_perceptron_dataset_ex_2"]["layers"], len(training_data_expected))

	mlp.train_weights(training_data, training_data_expected)

if __name__ == '__main__':
	inp = input("Select option:\n \
	\t1. Run Simple Perceptron to solve AND problem.\n \
	\t2. Run Simple Perceptron to solve XOR problem.\n \
	\t3. Run Non Lineal Perceptron with dataset from exercise 2.\n \
	\t4. Run Multi Layer Perceptron to solve XOR problem.\n \
	\t5. Run Multi Layer Perceptron to solve prime numbers problem.\n \
	\t6. Run Multi Layer Perceptron with dataset from exercise 2.\n")
	
	with open(current_path + "/config.json") as f:
            config = json.load(f)
	
	if inp == "1":
		simple_perceptron_ej1(AND, config)
	elif inp == "2":
		simple_perceptron_ej1(XOR, config)
	elif inp == "3":
		run_non_linear_for_exercise_two(config)
	elif inp == "4":
		multi_layer_xor(config)
	elif inp == "5":
		multi_layer_primos(config)
	elif inp == "6":
		multi_layer_ej2(config)
	else:
		print("Invalid option. Goodbye!")
