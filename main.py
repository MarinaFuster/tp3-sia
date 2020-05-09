from __future__ import print_function
import sys
import numpy as np
from simple_perceptron import train_weights

def main():

	iterations		= 100
	learning_rate  	= 0.5
	plot_iteration	= False
	stop_early 		= True

					# 	Bias 	i1 		i2 		y
	and_matrix = [	[1.00,	1.00,	1.00,	1.0],
					[1.00,	-1.00,	1.00,	-1.00],
					[1.00,	1.00,	-1.00,	-1.00],
					[1.00,	-1.00,	-1.00,	-1.00]]
	
					# 	Bias 	i1 		i2 		y
	xor_matrix = [	[1.00,	1.00,	1.00,	-1.0],
					[1.00,	-1.00,	1.00,	1.00],
					[1.00,	1.00,	-1.00,	1.00],
					[1.00,	-1.00,	-1.00,	-1.00]]

	weights= [	 0.1,	0.30,  0.80		] # initial weights specified in problem

	train_weights(and_matrix, weights=weights, iterations=iterations, learning_rate=learning_rate, plot_iteration=plot_iteration, stop_early=stop_early)


if __name__ == '__main__':
	main()