import sys
import numpy as np

def sigmoid(activation, beta=0.5):
	return 1 / (1 + np.exp(-2*beta*activation))

def sigmoid_derivate(activation, beta=0.5):
	return 2 * beta * sigmoid(activation, beta=beta) * (1 - sigmoid(activation, beta=beta))

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
def cost_function(matrix, weights, predict=sigmoid, derive=sigmoid_derivate, beta=0.5):
	error = 0.0
	for i in range(len(matrix)):
		activ_value = activation(matrix[i][:-1],weights)
		prediction = predict(activ_value, beta=beta)
		error+= (prediction - matrix[i][-1]) ** 2
	return error

# training algorithm for sample
def train_weights_nonlinear(matrix, weights, beta=0.5, predict=sigmoid, derive=sigmoid_derivate, epochs=100, learning_rate=1.00, stop_early=True):
	for epoch in range(epochs):
		epoch_cost = cost_function(matrix, weights, beta=beta)
		if epoch_cost < 0.001 and stop_early: break

		if epoch >= 10000:
			learning_rate = 0.1 
		
		# if epoch % 5000 == 0: print(epoch_cost)
		for i in range(len(matrix)):
			activ_value = activation(matrix[i][:-1], weights)
			prediction = predict(activ_value, beta=beta)  	   # get predicted classificaion
			derivative = derive(activ_value, beta=beta)	       # get derivative of inputs*weights
			error = matrix[i][-1] - prediction                 # get error from real classification
			for j in range(len(weights)):  	                   # calculate new weight for each nodes
				weights[j] = weights[j] + (learning_rate * error * derivative * matrix[i][j])  # update weights
	print("Broken in {} iterations with cost function {}".format(epoch, epoch_cost))
	return weights

def non_linear_perceptron_info(data_size, learning_rate, beta, epochs, selection_method, training_sample_size):
	r = ""
	r += "------ Non Linear Perceptron ----------\n"
	r += "Data collection size:{}\n".format(data_size)
	if selection_method == 0: r += "Selection method for training data: Random\n"
	elif selection_method == 1: r += "Selection method for training data: Sorted\n"
	r += "Training sample size:{}\n".format(training_sample_size)
	r += "Learning rate:{}\n".format(learning_rate)
	r += "Beta:{}\n".format(beta)
	r += "Maximum epoch:{}\n".format(epochs)
	return r

def test_perceptron(matrix, weights, predict=sigmoid, beta=0.5, print_results=False):
	predictions = []
	groundtruths = []
	RSEs = []
	if print_results:
		print('True values; Predicted values; Root Mean Errors')
	for i in range(len(matrix)):
		predictions.append(predict(activation(matrix[i][:-1], weights)))
		groundtruths.append(matrix[i][-1])
		RSEs.append(np.sqrt((predictions[i] - groundtruths[i])**2)) #root square error
		if print_results:
			print('%.5f		%.5f		%.5f' % (groundtruths[i], predictions[i], RSEs[i]))
	print("Cost function {} for testing sample".format(cost_function(matrix, weights, beta=beta)))
	return [predictions, groundtruths, RSEs]

