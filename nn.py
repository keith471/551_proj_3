

# We can represent the network as a bunch of layers where each layer has
# a weight matrix w giving the weights of edges leaving that layer
# an activation vector a giving the output leaving each node
# a bias vector giving the bias "weights" leaving from that layer and going to the next
# an activation function which describes how the input is turned into output

# the initial layer and output layers are a little different

# input layer
# needs an input vector x
#   then its output vector a is just x

# output layer
# needs an output vector y

import numpy as np
from numpy.random import rand


################################################################################
# Helpers
################################################################################

def rand(rows, cols):
    return (np.random(rows, cols) * 2 - 1) / 100


class FeedForwardNeuralNet(object):

    def __init__(self, m, layers, k, activation_func):
        '''initialize the neural network'''
        self.initialize(m, layers, k, activation_func)

    def initialize(self, m, layers, k, activation_func):
        '''
        initializes a network
        m (int): the number of features
        layers (list): the number of neurons in each hidden layer
        k (int): the number of classes
        activation (func): the activation function
        '''

        # input layer
        input_layer = InputLayer(m, layers[0])

        # hidden layers
        hidden_layers = []
        for i in range(len(layers) - 1):
            curr_layer_size = layers[i]
            next_layer_size = layers[i+1]
            hidden_layers.append(HiddenLayer(curr_layer_size, next_layer_size, activation_func))

        # add the last hidden layer
        hidden_layers.append(HiddenLayer(layers[len(layers)-1], k, activation_func))

        # output layer
        output_layer = OutputLayer(k, activation_func)

        # create a network
        self.network = NeuralNet(input_layer, hidden_layers, output_layer)


    def fprop(self, x):
        '''takes a training sample x and computes the activations in the network'''
        # the activation of the input layer



    def fit(self, X, y):
        # TODO

    def predict(self, X):
        # TODO


class HiddenLayer(object):

    def __init__(self, n, next_layer_size, activation):
        '''n is the number of neurons in this layer'''
        self.w = rand(next_layer_size, n)
        self.b = rand(1, next_layer_size)
        self.a = np.zeros((1, n))
        self.activation_func = activation_func

class InputLayer(object):

    def __init__(self, m, next_layer_size):
        self.w = rand(next_layer_size, m)
        self.a = np.zeros((1, m))
        self.b = rand(1, next_layer_size)

    def activate(self, x):
        '''activate the layer with an input x'''
        self.a = x

class OutputLayer(object):

    def __init__(self, k, activation_func):
        self.a = np.zeros((1, k))
        self.activation_func = activation_func

class NeuralNet(object):

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        # for convenience
        b =

    def activate(self, x):
        '''activates the network given an input x'''
        # activate the input layer
        self.input_layer.activate(x)

        # the activation of any given neuron j in layer l is b_(l-1)[j] + w_(l-1)[j]^T*a_(l-1)
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_layer.activate(curr_input)
