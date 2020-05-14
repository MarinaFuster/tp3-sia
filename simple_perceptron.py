import sys
import numpy as np
from graphics_plot import plot_and_or_xor

def step_function(inputs, weights):
	activation=0.0
	for inp,weight in zip(inputs, weights): # each input corresponding to each weight
		activation += inp*weight
	return 1.0 if activation>0.0 else -1.0

def sigmoid(inputs, weights):
	activation = 0.0
	for inp, weight in zip(inputs, weights):  # each input corresponding to each weight
		activation += inp * weight
	return 1 / (1 + np.exp(-activation))

def relu(inputs, weights):
	activation = 0.0
	for inp, weight in zip(inputs, weights):  # each input corresponding to each weight
		activation += inp * weight
	return activation if activation > 0.0 else 0.0

# calculates how well our algorithm is doing
# on classificating the training examples
def accuracy(matrix, weights, predict=step_function):
	num_correct = 0.0
	for i in range(len(matrix)):
		pred   = predict(matrix[i][:-1],weights) # get predicted classification
		if pred==matrix[i][-1]: num_correct+=1.0
	return num_correct/float(len(matrix))

# training algorithm for sample
def train_weights(matrix, weights, predict=step_function, iterations=100, learning_rate=1.00, plot=False, stop_early=True):
	for iteration in range(iterations):
		current_accuracy = accuracy(matrix, weights)

		if current_accuracy==1.0 and stop_early: break
        
		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1], weights) 				   # get predicted classificaion
			error      = matrix[i][-1]-prediction		 			 	   # get error from real classification
			for j in range(len(weights)): 				 				   # calculate new weight for each nodes
				weights[j] = weights[j]+(learning_rate*error*matrix[i][j]) # update weights

	if plot:
		plot_and_or_xor(matrix, predict, weights)
	return weights


def test_perceptron(matrix, weights, predict=relu, print_results=False):
	predictions = []
	groundtruths = []
	RSEs = []
	if print_results:
		print('True values; Predicted values; Root Mean Errors')
	for i in range(len(matrix)):
		predictions.append(np.round(predict(matrix[i][:-1], weights), 2))
		groundtruths.append(np.round(matrix[i][-1], 2))
		RSEs.append(np.round(np.sqrt((predictions[i] - groundtruths[i])**2), 2)) #root square error
		if print_results:
			print('%f		%f		%f' % (groundtruths[i], predictions[i], RSEs[i]))
	return [predictions, groundtruths, RSEs]