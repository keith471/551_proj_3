'''Feed-forward neural net implemenation'''

# Implemenation notes:
# This is a very 'mathematical' implementation in the sense that it does
# not attempt to actually construct a network, but rather represents the
# network using various vectors and matrices. To be precise, we
# can represent the network as a bunch of layers where each layer has
# a weight matrix w giving the weights of edges leaving that layer,
# an activation vector a giving the output leaving each node,
# and a bias vector b giving the biases leaving from that layer and going to the next.
# We note that the output layer has no bias vector (as each bias vector gives the biases
# for the NEXT layer, and there is no next layer to the output layer). We also note
# that the output from the first layer, a[0], is simply x (the current example).

from __future__ import print_function

import sys
import math
from collections import deque
from copy import deepcopy

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

def sigmoid_deriv(a):
    sa = sigmoid(a)
    return sa * (1 - sa)

def squared_error_loss(y_i, z_i, activation, activation_deriv):
    '''computes the partial derivative of the squared-error loss for the input scalar y_i
    and with respect to the input to the node z_i'''
    return (y_i - activation(z_i)) * (-1 * activation_deriv(z_i))

def cross_entroy_loss(y_i, z_i, activation, activation_deriv):
    '''computes the partial derivative of the cross-entropy loss for the input scalar y_i
    and with respect to the input to the node z_i'''
    return -1 * (y_i * (1 / activation(z_i)) * activation_deriv(z_i) + (1 - y_i) * (1 / (1 - activation(z_i))) * (-1 * activation_deriv(z_i)))

def squared_error(y, o):
    '''returns the mean squared error between the VECTORS y and o'''
    squared_diffs = [(y[i] - o[i])**2 for i in range(len(y))]
    sse = reduce(lambda x,y: x+y, squared_diffs, 0.0)
    mse = sse / len(y)
    return mse

def cross_entropy(y, o):
    '''returns the sum of the cross entropy between each element in the VECTORS y and o'''
    total = 0.0
    for i in range(len(y)):
        total += y[i] * np.log(o[i]) + (1.0 - y[i]) * np.log(1.0 - o[i])
    return total

class FeedForwardNeuralNet(object):

    def __init__(self, m, hidden_layer_sizes, k, alpha, lmda, activation_func=sigmoid, batch_size=1, verbose=False, max_iterations=1000):
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
        self.max_iterations = max_iterations
        self.initialize(m, hidden_layer_sizes, k, activation_func, verbose)

    def initialize(self, m, hidden_layer_sizes, k, activation_func, verbose):
        '''initializes a network'''
        self.network = NeuralNet(m, hidden_layer_sizes, k, activation_func, verbose=verbose)

    def fprop(self, x, y):
        '''takes a training sample x and computes the activations in the network
        y is only passed for the sake of comparison'''
        output = self.network.activate(x)
        '''
        if self.verbose:
            print('input:\t {0}'.format(x))
            print('act:\t {0}'.format(y))
            print('out:\t {0}'.format(output))
        '''

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

    def extract_actuals(self, y):
        '''takes an array 1-hot vectors and returns an array of indices of the 1s'''
        y_act = []
        for hot_vec in y:
            for i, v in enumerate(hot_vec):
                if v > 0:
                    y_act.append(i)
                    break
        return y_act

    def get_loss(self, act, pred):
        if len(act) != len(pred):
            print('ERROR: actual and predictions not of same length')
            sys.exit(1)
        loss = 0.0
        for i in range(len(act)):
            loss += squared_error(act[i], pred[i])
        if self.verbose:
            print('Example:')
            print('actual input:\t{0}'.format(act[len(act) - 1]))
            print('prediction:\t{0}'.format(pred[len(pred) - 1]))
        return loss / len(act)

    def get_error(self, act, pred):
        if len(act) != len(pred):
            print('ERROR: actual and predictions not of same length')
            sys.exit(1)
        incorrect = 0
        for i in range(len(act)):
            if act[i] != pred[i]:
                incorrect += 1
        return float(incorrect) / len(act)

    def get_performance(self, X, y):
        '''
        makes predictions for X and returns the loss and error against y
        '''
        y_hot = y
        y_act = self.extract_actuals(y)
        raw, cleaned = self.predict_raw(X)

        # compare y_hot to raw and y_act to cleaned
        loss = self.get_loss(y_hot, raw)
        error = self.get_error(y_act, cleaned)

        return loss, error

    def check_loss(self, tle, dle):
        '''
        looks through current and past losses to determine convergence, divergence, or overfitting
        and makes adjustments to alpha as need be
        returns
          1 if convergence
          -1 if divergence
          0 if more epochs are required
        '''
        # we want to continue as long as development loss is decreasing
        # if it increases, we divide alpha by two and continue
        #   if it increases for five consecutive rounds, then we have overfit
        # if if is steady, we divide alpha by two and continue
        #   if it remains steady or decreases for five consecutive iterations, then we have converged

        # convert to a more useful form
        tle = [v[0] for v in tle]
        dle = [v[0] for v in dle]

        if len(tle) <= 1:
            # we need to run at least two epochs to be able to check anything
            return

        # check for continuing divergence
        end = len(tle) - 1
        its = 6
        if len(tle) < its:
            # we need at least six epochs to determine divergence
            pass
        else:
            div = 0
            prev = dle[end - its + 1]
            for i in range(end - its + 2, end + 1):
                curr = dle[i]
                if curr > prev:
                    div += 1
                prev = curr
            if div == (its - 1):
                print('Performance has worsened for the past %d iterations despite decreases in alpha. Overfitting has begun.' % its)
                print('Final values of alpha and lambda: %.5f, %.5f' % (self.alpha, self.lmda))
                return -1

        # check for convergence
        its = 26
        if len(tle) < its:
            # we need at least 26 epochs to determine convergence
            pass
        else:
            conv = 0
            first = dle[end - its + 1]
            for i in range(end - its + 2, end + 1):
                curr = dle[i]
                if abs(curr - first) < 0.001:
                    conv += 1
            if conv == (its - 1):
                print('Peformance has remained steady for %d iterations. Gradient descent has converged.' % its)
                print('Final values of alpha and lambda: %.3f, %.3f' % (self.alpha, self.lmda))
                return 1

        if dle[end] > dle[end - 1]:
            self.alpha *= 0.9

        return 0

    def fit(self, X, y):
        '''
        fit the model by running forward and backward propagation for multiple epochs
        stop after convergence or divergence
        '''
        '''
        - [x] split into X_train, y_train, X_dev, y_dev
        - [x] for each point in X_train, run forward propagation and backwards propagation, and update delta_w and delta_b
        - [x] only update the parameters once per batch of points, resetting delta_w and delta_b each time you start a new batch
        - [x] after having run all the points and done the final update, predict each x in X_train and compare it to y_train
        to get the training loss
        - [x] measure the validation loss using X_dev and y_dev
        - [x] print a warning if validation error is less than training error, but continue
        - [x] continue until validation error increases or remains steady five times in a row - you have begun overfitting
        - [x] use the weights and biases obtained when lowest validation error was achieved (you will need to remember them)
        '''

        # training loss and error
        training_l_and_e = []

        # development loss and error
        dev_l_and_e = []

        # keep track of the last six configurations of the network in the event of overfitting
        # (since we will then have to recover a previous configuration)
        network_history = deque([])

        i = 0
        while True:
            # take a new train test split
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)

            # create batches for X_train and y_train based on self.batch_size
            batches = self.create_batches(X_train, y_train)

            # train the model using gradient descent
            for batch in batches:
                self.gradient_descent_iteration(batch)

            # save the latest network
            if len(network_history) == 6:
                network_history.popleft()
            network_history.append(deepcopy(self.network))

            # evaluate the loss and error
            training_loss, training_error = self.get_performance(X_train, y_train)
            dev_loss, dev_error = self.get_performance(X_dev, y_dev)

            training_l_and_e.append((training_loss, training_error))
            dev_l_and_e.append((dev_loss, dev_error))

            if self.verbose:
                print('-' * 80)
                print('Epoch %d' % i)
                print('_' * 80)

                print()
                print('training loss\ttraining error')
                print('%.6f\t%.6f' % (training_loss, training_error))
                print()

                print('dev loss\tdev error')
                print('%.6f\t%.6f' % (dev_loss, dev_error))
                print()

                if dev_loss < training_loss:
                    print('*' * 50)
                    print('WARNING: dev loss less than training loss')
                    print('*' * 50)
                    print()

            res = self.check_loss(training_l_and_e, dev_l_and_e)
            if res == 1:
                # convergence, and can maintain current network
                break
            elif res == -1:
                # divergence; reset network
                self.network = network_history[0]
                break

            i += 1

    def detect_stagnance(self, dle):
        dle = [v[0] for v in dle]
        if abs(dle[len(dle) - 2] - dle[len(dle) - 1]) < 0.001:
            return True
        return False

    def detect_oscillation(self, dle):
        dle = [v[0] for v in dle]
        end = len(dle) - 1
        its = 8
        if len(dle) < its:
            # we better run several epochs before deciding the error is oscillating
            return False
        else:
            count = 0
            prev = dle[end - 3]
            curr = dle[end - 2]
            diff = abs(prev - curr)
            prev = curr
            for i in range(end - 1, end + 1):
                curr = dle[i]
                if abs(curr - prev) > (0.01 * diff):
                    count += 1
            if count == 3:
                return True
            return False

    def tune_alpha_and_lmda(self, X, y):
        '''tunes the values of self.alpha and self.lmda'''
        # the goal here is to find a good starting point for alpha and lambda, working
        # from the initial values given
        # we will run gradient descent for 50 iterations
        # if the loss is not moving, we will increase it
        # if the loss is oscillating we will decrease it

        training_l_and_e = []
        dev_l_and_e = []

        for i in range(50):
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)

            batches = self.create_batches(X_train, y_train)

            for batch in batches:
                self.gradient_descent_iteration(batch)

            training_loss, training_error = self.get_performance(X_train, y_train)
            dev_loss, dev_error = self.get_performance(X_dev, y_dev)

            training_l_and_e.append((training_loss, training_error))
            dev_l_and_e.append((dev_loss, dev_error))

            if self.detect_oscillation(dev_l_and_e):
                self.alpha /= 2
            elif self.detect_stagnance(dev_l_and_e):
                self.alpha *= 2

            print('alpha: %.5f, lambda: %.5f' % (self.alpha, self.lmda))

    def get_max(self, output):
        '''returns the index of the output vector with the largest value'''
        max_i = 0
        max_v = 0.0
        for i, v in enumerate(output):
            if v > max_v:
                max_v = v
                max_i = i
        return max_i

    def predict_raw(self, X):
        '''returns raw as well as cleaned predictions'''
        raw = []
        cleaned = []
        for sample in X:
            o = self.network.activate(sample)
            raw.append(np.copy(o))
            cleaned.append(self.get_max(o))
        return raw, cleaned

    def predict(self, X):
        '''returns cleaned predictions'''
        return self.predict_raw[X][1]

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
    hidden_layer_sizes = [4,5]
    k = 2
    alpha = 20.0
    lmda = 0.0
    FFNN = FeedForwardNeuralNet(m, hidden_layer_sizes, k, alpha, lmda, verbose=True)
    FFNN.network.print_network()
    X = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    y = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    FFNN.fit(X, y)
