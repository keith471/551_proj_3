'''Feed-forward neural net implemenation'''

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

# TODO


from __future__ import print_function

import sys
import math

import numpy as np
from numpy.random import rand

################################################################################
# Helpers
################################################################################

def rand_list(size):
    return np.array(((np.random.rand(1, size) * 2 - 1) / 100)[0])

def rand_matrix(rows, cols):
    return (np.random.rand(rows, cols) * 2 - 1) / 100

def sigmoid(a):
    return 1 / (1 + math.exp(-a))

class FeedForwardNeuralNet(object):

    def __init__(self, m, hidden_layer_sizes, k, activation_func):
        '''
        initialize the neural network
            m (int): the number of features
            layers (list): the number of neurons in each hidden layer
            k (int): the number of classes
            activation (func): the activation function
        '''
        self.initialize(m, hidden_layer_sizes, k, activation_func)

    def initialize(self, m, hidden_layer_sizes, k, activation_func):
        '''initializes a network'''
        self.network = NeuralNet(m, hidden_layer_sizes, k, activation_func)

    def fprop(self, x):
        '''takes a training sample x and computes the activations in the network'''
        # the activation of the input layer
        self.network.activate(x)

    def bprop(self):
        # TODO
        pass

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass

class NeuralNet(object):

    def __init__(self, m, hidden_layer_sizes, k, activation_func):
        b = []
        w = []
        a = []
        # for convenience
        ls = [m] + hidden_layer_sizes + [k]

        for i in range(1, len(ls)):
            b.append(rand_list(ls[i]))

        for j in range(len(ls) - 1):
            w.append(rand_matrix(ls[j+1], ls[j]))

        for v in ls:
            a.append(np.zeros((1, v))[0])

        self.b = b
        self.w = w
        self.a = a
        self.activation_func = activation_func

    def activate(self, x):
        '''activates the network given an input x'''
        # activation of the first layer is simply x
        self.a[0] = np.array(x)
        # the activation of any given neuron j in layer l is
        # activation_func(b[l-1][j] + w[l-1][j]*a[l-1]^T)
        for l in range(1, len(self.a)):
            for j in range(len(self.a[l])):
                '''
                print('curr b:')
                print(self.b[l-1][j])
                print('curr w:')
                print(self.w[l-1][j])
                print('curr x:')
                print(self.a[l-1])
                print('dot prod:')
                print(np.dot(self.w[l-1][j], self.a[l-1]))
                sys.exit(1)
                '''
                self.a[l][j] = self.activation_func(self.b[l-1][j] + np.dot(self.w[l-1][j], self.a[l-1]))

    def print_network(self):
        print('a:')
        print(self.a)
        print()
        print('b:')
        print(self.b)
        print()
        print('w:')
        print(self.w)
        print()

if __name__ == '__main__':
    m = 3
    hidden_layer_sizes = [4]
    k = 2
    activation_func = sigmoid
    FFNN = FeedForwardNeuralNet(m, hidden_layer_sizes, k, activation_func)
    print('before')
    FFNN.network.print_network()
    x = [1.6, 2.4, 3.2]
    FFNN.fprop(x)
    print('after')
    FFNN.network.print_network()
