from simple_perceptron import step_function, sigmoid, accuracy
import numpy as np

def print_size(m, name):
    print("{}: {}x{}".format(name, len(m), len(m[0])))

class Multilayer_perceptron:

    def __init__(self, layers, data_attributes):
        # Layers is an array that contains the amount of layers (length), and the amount 
        # of nodes in each layer. For example:
        # layers => [3,2] means we have a hidden layer of size 3 and an output layer of size 2
        self.layers = layers
        self.weights = [None] * (len(layers))
        self.layer_outputs = [None] * (len(layers))
        self.layer_activations = [None] * (len(layers))
        self.deltas = [None] * (len(layers))

        self.learning_rate = 1

        # Initializing random weights
        # TODO: Check if random.random is a good weight initialization method
        for i in range(len(layers)):
            l_out = layers[i]
            l_in  = data_attributes if i == 0 else layers[i-1]
            self.weights[i] = np.random.rand(l_in,l_out)
        
        # print(self.weights)
    
    def train_weights(self, data, expected):
        n_epochs = 100
        for epoch in range(n_epochs):
            self.forward(data)        
            self.back_prop(data, expected)
        
    def predict(self, data):
        for i in range(len(self.layers)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]
            self.layer_outputs[i] = layer_input.dot(self.weights[i])            
            self.layer_activations[i] = self.sigmoid(self.layer_outputs[i])

        return self.layer_activations[len(self.layers) - 1] > 0.5

    # We expect data to already have the bias column added
    def forward(self, data):
        # Cicle through each layer
        for i in range(len(self.layers)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]
            self.layer_outputs[i] = layer_input.dot(self.weights[i])            
            self.layer_activations[i] = self.sigmoid(self.layer_outputs[i])

    def back_prop(self, data, expected):
        max_layer = len(self.layers) - 1
        sum_error = expected - self.layer_activations[max_layer]
        for i in range(max_layer, -1, -1):
            # Casp especial para ultima y para primer layer:
            prev_activation = data if i == 0 else self.layer_activations[i - 1]

            if i == max_layer:
                self.deltas[i] = -1 * sum_error * self.sigmoid_derivate(self.layer_outputs[i])
            else:
                self.deltas[i] = self.sigmoid_derivate(self.layer_outputs[i]) * (self.deltas[i+1].dot(self.weights[i+1].transpose()))
            deltaW = self.learning_rate * np.transpose(prev_activation).dot(self.deltas[i])
            self.weights[i] = self.weights[i] - deltaW

    def sigmoid(self, layer_output):
        return 1 / (1 + np.exp(-layer_output))

    def sigmoid_derivate(self, m):
        return self.sigmoid(m) * (1- self.sigmoid(m))

            
# Forward propagation


# Training data:
# size NxM where N is the amount of training samples and M is the amount of attributes per sample

# Hidden layer of size 3
# Weights: Each unit in the layer has an array of weights for each attribute of the previous layer's output
# Weights size: Mx3
# Output size: Nx3

# Output layer of size 1
# Weights size: 3x1
# Output: size Nx1

# Back propagation
