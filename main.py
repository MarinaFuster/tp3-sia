from __future__ import print_function
import sys
import numpy as np
from simple_perceptron import train_weights, sigmoid
from multilayer_perceptron import Multilayer_perceptron

def main():

	iterations		= 100
	learning_rate  	= 0.5
	plot_iteration	= True
	stop_early 		= True

					# 	Bias 	x1 		x2 		y
	and_matrix = [	[1.00,	1.00,	1.00,	1.0],
					[1.00,	-1.00,	1.00,	-1.00],
					[1.00,	1.00,	-1.00,	-1.00],
					[1.00,	-1.00,	-1.00,	-1.00]]
	
					# 	Bias 	x1 		x2 		y
	xor_matrix = [	[1.00,	1.00,	1.00,	-1.0],
					[1.00,	-1.00,	1.00,	1.00],
					[1.00,	1.00,	-1.00,	1.00],
					[1.00,	-1.00,	-1.00,	-1.00]]

	weights= [	 0.1,	0.30,  0.80		] # initial weights

	train_weights(and_matrix, weights=weights, predict=sigmoid, iterations=iterations, learning_rate=learning_rate, plot_iteration=plot_iteration, stop_early=stop_early)


	


def multi_layer():
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

if __name__ == '__main__':
	main()
	multi_layer()	
