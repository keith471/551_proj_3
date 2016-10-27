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
from sklearn.model_selection import train_test_split

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

    def __init__(self, m, hidden_layer_sizes, k, alpha, lmda, activation_func=sigmoid, batch_size=1, verbose=False):
        '''
        initialize the neural network
            m (int): the number of features
            layers (list): the number of neurons in each hidden layer
            k (int): the number of classes
            activation (func): the activation function
        '''
        self.alpha = alpha
        self.lmda = lmda
        self.batch_size = batch_size
        self.verbose = verbose
        self.initialize(m, hidden_layer_sizes, k, activation_func, verbose)

    def initialize(self, m, hidden_layer_sizes, k, activation_func, verbose):
        '''initializes a network'''
        self.network = NeuralNet(m, hidden_layer_sizes, k, activation_func, verbose=verbose)

    def fprop(self, x, y):
        '''takes a training sample x and computes the activations in the network
        y is only passed for the sake of comparison'''
        output = self.network.activate(x)
        if self.verbose:
            #print('input:\t {0}'.format(x))
            print('act:\t {0}'.format(y))
            print('out:\t {0}'.format(output))

    def bprop(self, y):
        self.network.compute_deltas(y)
        self.network.compute_partials()
        self.network.update_deltas()

    def gradient_descent_iteration(self, batch):
        '''contains all the steps in one iteration of gradient descent
        batch is an array of (x,y) pairs'''
        # run forward and backward propagation for the entire batch,
        # updating delta_w and delta_b for each pair (x, y) in the batch
        for v in batch:
            x, y = v
            self.fprop(x, y)
            self.bprop(y)

        # update the parameters w and b based on the results of the batch
        self.network.update_params(self.alpha, self.lmda, len(batch))

    def create_batches(self, X_train, y_train):
        num_batches = len(X_train) / self.batch_size
        if len(X_train) % num_batches != 0:
            num_batches += 1
        batches = [[] for i in range(num_batches)]
        for i, x in enumerate(X_train):
            y = y_train[i]
            batches[i % num_batches].append((x, y))
        return batches

    def get_performance(self, X, y):
        '''
        makes predictions for X and returns the loss and error against y
        '''
        # TODO implement this
        # for each output
        loss = -(y * log(o) + (1-y) * log(1-o))
        # then implement multiple epochs and see if you can converge
        pass


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
        # should all be done in a loop

        # take a new train test split
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)

        # create batches for X_train and y_train based on self.batch_size
        batches = self.create_batches(X_train, y_train)

        for batch in batches:
            self.gradient_descent_iteration(batch)

        training_loss, training_error = get_performance(X_train, y_train)
        dev_loss, dev_error = get_performance(X_dev, y_dev)

        print()
        print('training loss\ttraining error')
        print('%.3f\t%.3f' % (training_loss, training_error))
        print()

        print('dev loss\tdev error')
        print('%.3f\t%.3f' % (dev_loss, dev_error))
        print()

        if dev_loss < training_loss:
            print('!' * 40)
            print('WARNING: dev loss less than training loss')
            print('_' * 40)
            print()

        print('completed one epoch')
        print()


    def predict(self, X):
        '''simply pass each x through the network using activate to get the output
        process the output to get the most probable class
        record in an array
        return array of the predictions for each x'''
        pass

class NeuralNet(object):

    def __init__(self, m, hidden_layer_sizes, k, activation_func, verbose=False):
        b = []  # bias terms
        w = []  # weights
        a = []  # outputs
        d = []  # neuron deltas
        delta_w = []    # stores the changes in w for a batch of examples (one iteration of gradient descent)
        delta_b = []    # stores the changes in b " " " ...

        # for convenience, store the size of each layer in a single list
        ls = [m] + hidden_layer_sizes + [k]

        for i in range(1, len(ls)):
            b.append(rand_list(ls[i]))
            delta_b.append(np.zeros((1, ls[i]))[0])

        for j in range(len(ls) - 1):
            w.append(rand_matrix(ls[j+1], ls[j]))
            delta_w.append(np.zeros((ls[j+1], ls[j])))

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
        self.delta_w = delta_w
        self.delta_b = delta_b
        self.activation_func = activation_func
        self.verbose = verbose

    def activate(self, x):
        '''activates the network given an input x
        returns the output'''
        # activation of the first layer is simply x
        self.a[0] = np.array(x)
        # the activation of any given neuron j in layer l is
        # activation_func(b[l-1][j] + w[l-1][j]*a[l-1]^T)
        for l in range(1, len(self.a)):
            for j in range(len(self.a[l])):
                self.a[l][j] = self.activation_func(self.b[l-1][j] + self.w[l-1][j].dot(self.a[l-1]))

        return self.a[len(self.ls) - 1]

    def differentiate_layer_output(self, l):
        '''assumes sigmoid activation'''
        # TODO make generic
        return [self.a[l][i]*(1 - self.a[l][i]) for i in range(self.ls[l])]


    def compute_deltas(self, y):
        '''y is a 1-hot vector'''
        # for each output unit i in the output layer n, set d[n][i] =
        n = len(self.ls) - 1
        f_prime = self.differentiate_layer_output(n)
        self.d[n] = np.multiply(-(y - self.a[n]), f_prime)

        # for each node i in layer l, set d[l][i] =
        for l in reversed(range(1, n)):
            f_prime = self.differentiate_layer_output(l)
            self.d[l] = np.multiply(self.w[l].T.dot(self.d[l+1].reshape((len(self.d[l+1]),1))).T, f_prime)[0]

    def compute_partials(self):
        self.w_partials = [self.d[l+1].reshape((len(self.d[l+1]),1)).dot(self.a[l].reshape((len(self.a[l]),1)).T) for l in range(len(self.ls) - 1)]
        self.b_partials = [self.d[l+1] for l in range(len(self.ls) - 1)]

    def update_deltas(self):
        for l in range(len(self.ls) - 1):
            self.delta_w[l] += self.w_partials[l]
            self.delta_b[l] += self.b_partials[l]

    def reset_deltas(self):
        '''reset delta_w and delta_b to all zeros
        called after each iteration of gradient descent'''
        delta_w = []
        delta_b = []

        for i in range(1, len(self.ls)):
            delta_b.append(np.zeros((1, self.ls[i]))[0])

        for j in range(len(self.ls) - 1):
            delta_w.append(np.zeros((self.ls[j+1], self.ls[j])))

        self.delta_w = delta_w
        self.delta_b = delta_b

    def update_params(self, alpha, lmda, batch_size):
        '''batch_size is the number of points in the batch'''
        print('updating params')
        print()
        # update the parameters
        for l in range(len(self.ls) - 1):
            self.w[l] -= alpha * ((self.delta_w[l] / batch_size) + (lmda * self.w[l]))
            self.b[l] -= alpha * (self.delta_b[l] / batch_size)
        #if self.verbose:
            #self.print_network()
        self.reset_deltas()

    def print_element(self, element, rng, offset=0):
        for i in rng:
            print('Layer %d' % (i + offset))
            print(element[i + offset])

    def print_network(self):
        print('-' * 80)
        print('NETWORK')
        print('_' * 80)
        print()
        print('a:')
        self.print_element(self.a, range(len(self.ls)))
        print()
        print('b:')
        self.print_element(self.b, range(len(self.ls) - 1))
        print()
        print('delta_b:')
        self.print_element(self.delta_b, range(len(self.ls) - 1))
        print()
        print('w:')
        self.print_element(self.w, range(len(self.ls) - 1))
        print()
        print('delta_w:')
        self.print_element(self.delta_w, range(len(self.ls) - 1))
        print()
        print('d:')
        self.print_element(self.d, range(len(self.ls) - 1), 1)
        print()
        print('~' * 80)
        print('END OF NETWORK')
        print('_' * 80)

if __name__ == '__main__':
    m = 3
    hidden_layer_sizes = [4, 5]
    k = 2
    alpha = 2
    lmda = 0
    FFNN = FeedForwardNeuralNet(m, hidden_layer_sizes, k, alpha, lmda, verbose=True)
    FFNN.network.print_network()
    X = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    y = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    FFNN.fit(X, y)
