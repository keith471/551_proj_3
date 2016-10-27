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

    def __init__(self, m, hidden_layer_sizes, k, activation_func=sigmoid, batch_size=1):
        '''
        initialize the neural network
            m (int): the number of features
            layers (list): the number of neurons in each hidden layer
            k (int): the number of classes
            activation (func): the activation function
        '''
        self.batch_size = batch_size
        self.initialize(m, hidden_layer_sizes, k, activation_func)

    def initialize(self, m, hidden_layer_sizes, k, activation_func):
        '''initializes a network'''
        self.network = NeuralNet(m, hidden_layer_sizes, k, activation_func)

    def fprop(self, x):
        '''takes a training sample x and computes the activations in the network'''
        self.network.activate(x)

    def bprop(self):
        self.network.compute_deltas(y)
        self.network.compute_partials()

    def fit(self, X, y):
        '''
        - split into X_train, y_train, X_dev, y_dev
        - for each point in X_train, run forward propagation and backwards propagation, and update
        delta_w and delta_b
        - only update the parameters once per batch of points, resetting delta_w and delta_b each time you start a new batch
        - after having run all the points and done the final update, predict each x in X_train and compare it to y_train
        to get the training loss
        - measure the validation loss using X_dev and y_dev
        - print a warning if validation error is less than training error, but continue
        - continue until validation error increases two times in a row - you have begun overfitting
        - use the weights and biases obtained when lowest validation error was achieved (you will need to remember them)
        '''
        pass

    def predict(self, X):
        '''simply pass each x through the network using activate to get the output
        process the output to get the most probable class
        record in an array
        return array of the predictions for each x'''
        pass

class NeuralNet(object):

    def __init__(self, m, hidden_layer_sizes, k, activation_func):
        b = []  # bias terms
        w = []  # weights
        a = []  # outputs
        d = []  # deltas

        # for convenience, store the size of each layer in a single list
        ls = [m] + hidden_layer_sizes + [k]

        for i in range(1, len(ls)):
            b.append(rand_list(ls[i]))

        for j in range(len(ls) - 1):
            w.append(rand_matrix(ls[j+1], ls[j]))

        for v in ls:
            a.append(np.zeros((1, v))[0])
            d.append(np.zeros((1, v))[0])

        self.ls = ls
        self.m = m
        self.k = k
        self.b = b
        self.w = w
        self.a = a
        self.d = d
        self.delta_w = # zeros
        self.delta_b = # zeros
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
                self.a[l][j] = self.activation_func(self.b[l-1][j] + self.w[l-1][j].dot(self.a[l-1])))

    def differentiate_layer_output(self, l):
        '''assumes sigmoid activation'''
        # TODO make generic
        return [self.a[l][i]*(1 - self.a[l][i]) for i in range(self.ls[l])]


    def delta_out(self, y):
        '''y is a 1-hot vector'''
        # for each output unit i in the output layer n, set d[n][i] =
        n = self.layer_count - 1
        f_prime = self.differentiate_layer_output(n)
        self.d[n] = np.multiply(-(y - self.a[n]), f_prime)

    def delta_rest(self):
        # for each node i in layer l, set d[l][i] =
        for l in reversed(range(1, self.layer_count - 1)):
            f_prime = self.differentiate_layer_output(l)
            self.d[l] = np.multiply(self.w[l].T.dot(self.d[l+1].reshape((3,1))), f_prime)

    def compute_partials(self):
        self.w_partials = [self.d[l+1].reshape((3,1)).dot(self.a[l].reshape((3,1)).T) for l in range(len(self.ls) - 1)]
        self.b_partials = [self.d[l+1] for l in range(len(self.ls))]

    def
        for l in range(len(ls) - 1):
            self.delta_w[l] += self.w_partials[l]
            self.delta_b[l] += self.b_partials[l]

    def update_params(self):
        '''m is the number of points in the batch'''

        # update the parameters
        for l in range(len(ls) - 1):
            self.w[l] -= alpha * ((self.delta_w[l] / m) + (lmda * self.w[l]))
            self.b[l] -= alpha * (self.delta_b[l] / m)

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
