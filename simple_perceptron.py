import sys
import numpy as np
from graphics_plot import plot_and_or_xor

def step_function(inputs, weights):
	activation=0.0
	for inp,weight in zip(inputs, weights): # each input corresponding to each weight
		activation += inp*weight
	return 1.0 if activation >= 0.0 else -1.0

# calculates how well our algorithm is doing
# on classificating the training examples
def accuracy(matrix, weights, predict=step_function):
	num_correct = 0.0
	for i in range(len(matrix)):
		pred   = predict(matrix[i][:-1],weights) # get predicted classification
		if pred==matrix[i][-1]: num_correct+=1.0
	return num_correct/float(len(matrix))

# training algorithm for sample
def train_weights(matrix, weights, predict=step_function, epochs=100, learning_rate=1.00, plot=False, stop_early=True):

	# Adds a column of ones, so that w0 becomes the weight representing the bias
	matrix = np.column_stack((np.ones((len(matrix),1)), matrix))

	for epoch in range(epochs):
		current_accuracy = accuracy(matrix, weights)

		# We do not need further training if we reached our max accuracy
		if current_accuracy==1.0 and stop_early: break
        
		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1], weights) 				   # get predicted classificaion
			error      = matrix[i][-1]-prediction		 			 	   # get error from real classification
			for j in range(len(weights)):								   # calculate new weight for each nodes
				weights[j] = weights[j]+(learning_rate*error*matrix[i][j]) # update weights w = w + deltaw

	print("Broken in {} iterations with accuracy  {}".format(epoch, current_accuracy))
	if plot:
		plot_and_or_xor(matrix, predict, weights)
	return weights

def simple_perceptron_info(learning_rate, epochs, problem):
	r = ""
	r += "------ Simple Perceptron ----------\n"
	if problem == 0: r += "Problem:AND\n"
	elif problem == 1: r += "Problem:XOR\n"
	r += "Learning rate:{}\n".format(learning_rate)
	r += "Maximum epoch:{}\n".format(epochs)
	return r
