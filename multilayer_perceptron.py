import numpy as np
import pandas as pd

def sigmoid(input, weights, beta=0.05):
    activation = 0.0
    for inp, weight in zip(inputs, weights):  # each input corresponding to each weight
        activation += inp * weight
    return 1 / (1 + np.exp(-2*beta*activation))

def sigmoid_derivate(input, weights, beta=0.05):
    return 2 * beta * sigmoid(input, weights) * (1 - sigmoid(input, weights))

def print_size(m, name):
    print("{}: {}x{}".format(name, len(m), len(m[0])))

def import_numdata():
    df = pd.read_csv("data/numeros.txt", header = None, delimiter=' ')
    df = df.drop(columns=5)
    data = df.to_numpy()
    data_stacked = data.reshape(10,35)

    #--- to inspect correct import of data enter the desired number in target to see a print of the given row reshaped to an image
    #target = 9
    #stacked = data_stacked[target,:]
    #image = stacked.reshape(7,5)
    #print("target = %d" % target)
    #print(image)
    #---endof inspection

    return data_stacked

class Multilayer_perceptron:

    def __init__(self, layers, data_attributes):
        # Initialization of arrays because I dont know python
        self.layers = layers
        self.weights = [None] * (len(layers))
        self.bias = [None] * (len(layers))
        self.layer_outputs = [None] * (len(layers))
        self.layer_activations = [None] * (len(layers))
        self.deltas = [None] * (len(layers))
        self.deltaW = [None] * (len(layers))


        # Network constants
        self.learning_rate = 1.3
        self.beta = 0.5
        self.max_epochs = 1000000
        self.error_threshold = 1e-5
        self.batch = False
        # Initializing random weights
        # TODO: Check if random.random is a good weight initialization method
        for i in range(len(layers)):
            l_out = layers[i]
            l_in  = data_attributes if i == 0 else layers[i-1]
            self.weights[i] = np.random.randn(l_in+1,l_out) * 0.01
            self.bias[i] = np.ones((l_out,1))

    
    def train_weights(self, data, expected):
        print(self)
        self.error = self.error_threshold + 1
        for epoch in range(self.max_epochs):
            if self.batch:
                self.forward(row)        
                self.back_prop(row, result)
            else:
                for sample,result in zip(data,expected):
                    row = np.array([sample])
                    self.forward(row)        
                    self.back_prop(row, result)

            if self.error < self.error_threshold:
                print("Broken in {} iterations because error is {}".format(epoch, self.error_threshold))
                break

        print("ERROR: {}".format(self.error))

    def __str__(self):
        r = ""
        r += "------ Multi layer perceptron ----------\n"
        for i in range(len(self.layers)):
            if i == len(self.layers) -1:
                r += "Output layer size:{}\n".format(self.layers[i])
            else:
                r += "Hidden layer {} size:{}\n".format(i, self.layers[i])
        
        r += "Beta:{}\n".format(self.beta)
        r += "Learning rate:{}\n".format(self.learning_rate)
        r += "Maximum epoch:{}\n".format(self.max_epochs)
        r += "Error threshold:{}\n".format(self.error_threshold)
        return r
        
    def predict(self, data):
        for i in range(len(self.layers)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]
            layer_input = np.column_stack((layer_input, np.ones((len(layer_input),1))))
            self.layer_outputs[i] = layer_input.dot(self.weights[i])
            self.layer_activations[i] = self.sigmoid(self.layer_outputs[i])

        print(self.layer_activations[len(self.layer_activations) - 1])
        return self.layer_activations[len(self.layers) - 1] > 0.5



    # We expect data to already have the bias column added
    def forward(self, data):
        # Cicle through each layer
        for i in range(len(self.layers)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]

            layer_input = np.column_stack((layer_input, np.ones((len(layer_input),1))))
            self.layer_outputs[i] = layer_input.dot(self.weights[i]) + self.bias[i].T
            self.layer_activations[i] = self.sigmoid(self.layer_outputs[i])
        

    def back_prop(self, data, expected):
        output_layer = len(self.layers) - 1
        error_vector = (expected - self.layer_activations[output_layer])
        self.error = np.sum(error_vector**2)
        for i in range(output_layer, -1, -1):
            # Casp especial para ultima y para primer layer:

            
            curr_output = self.layer_outputs[i]

            if i == output_layer:
                self.deltas[i] = error_vector * self.sigmoid_derivate(self.layer_outputs[i])

            else:
                dt = (self.deltas[i+1].dot(self.weights[i+1].T))
                # print_size(dt[:,:-1], "dt")
                # print_size(curr_output, "curr_output")
                self.deltas[i] = self.sigmoid_derivate(curr_output) * dt[:,:-1]


        for i in range(len(self.layers)):
            prev_activation = data if i == 0 else self.layer_activations[i - 1]
            prev_activation = np.column_stack((prev_activation, np.ones((len(prev_activation),1))))
            self.deltaW[i] = self.learning_rate * ((prev_activation).T).dot(self.deltas[i])

        # Actualizar los pesos
        for i in range(output_layer, -1, -1):
            self.weights[i] = self.weights[i] + self.deltaW[i]
            # self.bias[i] = self.bias[i] + self.deltaB[i]

    def sigmoid(self, layer_output):
        return 1 / (1 + np.exp(-2*self.beta*layer_output))

    def sigmoid_derivate(self, m):
        return 2 * self.beta * (self.sigmoid(m) * (1 - self.sigmoid(m)))