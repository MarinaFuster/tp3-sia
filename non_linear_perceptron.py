import sys
import numpy as np
from simple_perceptron import accuracy

def sigmoid(activation, beta=0.05):
	return 1 / (1 + np.exp(-2*beta*activation))

def sigmoid_derivate(activation, beta=0.05):
	return 2 * beta * sigmoid(activation) * (1 - sigmoid(activation))

def activation(inputs, weights):
	activation = 0.0
	for inp, weight in zip(inputs, weights):  # each input corresponding to each weight
		activation += inp * weight
	return activation

def tanh_activation(inputs, weights, L=100):
	activation = 0.0
	for inp, weight in zip(inputs, weights):  # each input corresponding to each weight
		activation += inp * weight
	return np.tanh(activation)

# calculates how well our algorithm is doing
def accuracy(matrix, weights, predict=sigmoid, derive=sigmoid_derivate, beta=0.05):
	error = 0.0
	for i in range(len(matrix)):
		activ_value = activation(matrix[i][:-1],weights)
		prediction = predict(activ_value, beta)
		error+= (prediction - matrix[i][-1]) ** 2
	return error

# training algorithm for sample
def train_weights_nonlinear(matrix, weights, beta=0.05, predict=sigmoid, derive=sigmoid_derivate, epochs=100, learning_rate=1.00, stop_early=True):
	for epoch in range(epochs):

		epoch_accuracy = accuracy(matrix, weights, beta=beta)
		if epoch_accuracy < 0.001 and stop_early: break
		
		for i in range(len(matrix)):
			activ_value = activation(matrix[i][:-1], weights)
			prediction = predict(activ_value, beta=beta)  	   # get predicted classificaion
			derivative = derive(activ_value, beta=beta)	       # get derivative of inputs*weights
			error = matrix[i][-1] - prediction                 # get error from real classification
			for j in range(len(weights)):  	                   # calculate new weight for each nodes
				weights[j] = weights[j] + (learning_rate * error * derivative * matrix[i][j])  # update weights

	return weights

# def test_perceptron(matrix, weights, predict=sigmoid, print_results=False):
# 	predictions = []
# 	groundtruths = []
# 	RSEs = []
# 	if print_results:
# 		print('True values; Predicted values; Root Mean Errors')
# 	for i in range(len(matrix)):
# 		predictions.append(np.round(predict(matrix[i][:-1], weights), 2))
# 		groundtruths.append(np.round(matrix[i][-1], 2))
# 		RSEs.append(np.round(np.sqrt((predictions[i] - groundtruths[i])**2), 2)) #root square error
# 		if print_results:
# 			print('%f		%f		%f' % (groundtruths[i], predictions[i], RSEs[i]))
# 	return [predictions, groundtruths, RSEs]