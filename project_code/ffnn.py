'''Feed-forward neural net implemenation'''

# Implemenation notes:
# This is a very 'mathematical' implementation in the sense that it does
# not attempt to actually construct a network, but rather represents the
# network using various vectors and matrices.

from __future__ import print_function

import sys
import math
from collections import deque
from copy import deepcopy
import cPickle as pickle

import numpy as np
from numpy.random import RandomState
from numpy.random import rand

from time import time

from sklearn.model_selection import train_test_split

from postprocess import write_errs_to_csv

################################################################################
# Helpers
################################################################################

def rand_matrix(n_out, n_in):
    '''returns an np array of size size initialized with random numbers'''
    rng = RandomState()
    return rng.uniform(
        low=-4 * np.sqrt(6. / (n_in + n_out)),
        high=4 * np.sqrt(6. / (n_in + n_out)),
        size=(n_out, n_in)
    )

def squared_error_loss(y, o, f_prime):
    return np.multiply(-(y - o), f_prime)

def cross_entropy_loss(y, o, f_prime):
    y = np.array(y)
    f_prime = np.array(f_prime)
    return -1 * (np.multiply(np.multiply(y, 1/o), f_prime) + np.multiply(np.multiply((1 - y), (1/(1 - o))), (-1 * f_prime)))

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

class SigmoidActivator(object):

    def activate(self, a):
        # if statements ensure we avoid overflow issues
        if a > 60:
            return 1.
        if a < -60:
            return 0.
        return 1 / (1 + math.exp(-a))

    def deriv(self, a):
        sa = sigmoid(a)
        return sa * (1 - sa)

    def deriv_layer(self, o):
        '''o is the output vector for the layer'''
        return [o[i] * (1 - o[i]) for i in range(len(o))]

class FeedForwardNeuralNet(object):

    def __init__(self, m, hidden_layer_sizes, k, alpha, lmda, n_epochs, activator=SigmoidActivator, loss_function=cross_entropy_loss, batch_size=100, verbose=False):
        '''
        initialize the neural network
            m (int): the number of features
            layers (list): the number of neurons in each hidden layer
            k (int): the number of classes
            alpha (float): the learning rate
            lambda (float): Ridge regularization penalty
            n_epochs (int): the number of epochs to train for
            activator (class): the activator, defining the activation function to use and its derivative
            loss_function (function): the loss function to use
            batch_size (int): the number of examples to perform gradient descent on before updating the weights and biases of the network
            verbose (boolean): whether or not to print out helpful information as we go
        '''
        self.alpha = alpha
        self.lmda = lmda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.initialize(m, hidden_layer_sizes, k, activator, loss_function, verbose)

    def pretty_print(self):
        self.network.pretty_print()
        print('Alpha:\t%f' % self.alpha)
        print('Lambda:\t%f' % self.lmda)

    def initialize(self, m, hidden_layer_sizes, k, activator, loss_function, verbose):
        '''initializes a network'''
        self.network = NeuralNet(m, hidden_layer_sizes, k, activator, loss_function, verbose=verbose)

    def fprop(self, x, y):
        '''takes a training sample x and computes the activations in the network
        y is only passed for the sake of comparison'''
        output = self.network.activate(x)

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
        if num_batches == 0 or len(X_train) % num_batches != 0:
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

    def fit(self, X, y, use_test_set=False, pickle_best=False, write_errs=False):

        # take a train/dev/test split
        if use_test_set:
            X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.10)
            X_train, X_dev, y_train, y_dev = train_test_split(X_rest, y_rest, test_size=0.25)
        else:
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.30)

        # create batches for X_train and y_train based on self.batch_size
        batches = self.create_batches(X_train, y_train)
        n_train_batches = len(batches)

        # keep track of training/dev loss and error
        training_l_and_e = []
        dev_l_and_e = []

        # early-stopping parameters
        # look as this many examples regardless
        patience = 10000

        # wait this much longer when a new best is found
        patience_increase = 2

        # a relative improvement of this much is considered significant
        improvement_threshold = 0.995

         # go through this many minibatchs before checking the network on the
         # validation set; in this case we check every epoch
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = np.inf
        best_validation_err = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time()

        epoch = 0
        done_looping = False
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            if self.verbose:
                print('-' * 30)
                print('Epoch %d' % epoch)
                print('_' * 30)
            # train the model using gradient descent
            for minibatch_index, batch in enumerate(batches):
                self.gradient_descent_iteration(batch)

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    training_loss, training_error = self.get_performance(X_train, y_train)
                    dev_loss, dev_error = self.get_performance(X_dev, y_dev)

                    training_l_and_e.append((training_loss, training_error))
                    dev_l_and_e.append((dev_loss, dev_error))

                    if self.verbose:
                        print(
                            'epoch %i, minibatch %i/%i, validation (dev) error %f %%' %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                dev_error * 100.
                            )
                        )
                        print()
                        print('training loss\ttraining error')
                        print('%.6f\t%.6f' % (training_loss, training_error))
                        print()

                        print('dev loss\tdev error')
                        print('%.6f\t%.6f' % (dev_loss, dev_error))
                        print()

                    # if we got the best validation score until now
                    if dev_loss < best_validation_loss:
                        print('achieved new validation best: %f (loss) %f (error)' % (dev_loss, dev_error))
                        # improve patience if error improvement is good enough
                        if dev_error < (best_validation_err * improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = dev_loss
                        best_validation_err = dev_error
                        best_iter = iter

                        # test it on the test set
                        if use_test_set:
                            test_loss, test_error = self.get_performance(X_test, y_test)

                            if self.verbose:
                                print(('     epoch %i, minibatch %i/%i, test error of '
                                       'best model %f %%') %
                                      (epoch, minibatch_index + 1, n_train_batches,
                                       test_error * 100.))

                        # save the best model
                        if pickle_best:
                            print('pickling model')
                            print()
                            with open('best_model.pkl', 'wb') as f:
                                pickle.dump(self.network, f)

                if patience <= iter:
                    done_looping = True
                    break

        # save the training and validation loss/error history to disk
        if write_errs:
            write_errs_to_csv(training_l_and_e, dev_l_and_e)

        end_time = time()
        if self.verbose:
            print('-' * 40)
            print('Optimization complete')
            print('_' * 40)
            if use_test_set:
                print(('Best validation score of %f %% '
                       'obtained at iteration %i, with test performance %f %%') %
                      (best_validation_err * 100., best_iter + 1, test_error * 100.))
            else:
                print(('Best validation loss of %f %% '
                       'obtained at iteration %i') %
                      (best_validation_err * 100., best_iter + 1))
            print('Total training time: %.2fm' % ((end_time - start_time) / 60.))

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

    def tune_alpha(self, X, y):
        '''tunes the starting value of the learning rate'''
        # the goal here is to find a good starting point for alpha, working
        # from the initial value given
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
                self.alpha /= 2.
            elif self.detect_stagnance(dev_l_and_e):
                self.alpha *= 2.

            print('alpha: %.5f' % self.alpha)

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
        return self.predict_raw(X)[1]

class NeuralNet(object):
    '''The heart of the FeedForwardNeuralNet'''

    def __init__(self, m, hidden_layer_sizes, k, activator, loss_function, verbose=False):
        b = []  # bias terms
        w = []  # weights
        a = []  # outputs
        d = []  # neuron deltas
        delta_w = []    # stores the changes in w for a batch of examples (one iteration of gradient descent)
        delta_b = []    # stores the changes in b " " " ...

        # for convenience, store the size of each layer in a single list
        ls = [m] + hidden_layer_sizes + [k]

        for i in range(1, len(ls)):
            b.append(np.zeros((ls[i],)))
            delta_b.append(np.zeros((ls[i],)))

        for j in range(len(ls) - 1):
            w.append(rand_matrix(ls[j+1], ls[j]))
            delta_w.append(np.zeros((ls[j+1], ls[j])))

        for v in ls:
            a.append(np.zeros((v,)))
            d.append(np.zeros((v,)))

        self.ls = ls
        self.m = m
        self.k = k
        self.b = b
        self.w = w
        self.a = a
        self.d = d
        self.delta_w = delta_w
        self.delta_b = delta_b
        self.activator = activator()
        self.loss_function = loss_function
        self.verbose = verbose

    def activate(self, x):
        '''activates the network given an input x
        returns the output'''
        # activation of the first layer is simply x
        self.a[0] = np.array(x)
        # the activation of any given neuron j in layer l is
        # activator.activate((b[l-1][j] + w[l-1][j]*a[l-1]^T))
        for l in range(1, len(self.a)):
            for j in range(len(self.a[l])):
                self.a[l][j] = self.activator.activate((self.b[l-1][j] + self.w[l-1][j].dot(self.a[l-1])))

        return self.a[len(self.ls) - 1]

    def compute_deltas(self, y):
        '''y is a 1-hot vector'''
        # output layer
        n = len(self.ls) - 1
        f_prime = self.activator.deriv_layer(self.a[n])
        self.d[n] = self.loss_function(y, self.a[n], f_prime)

        # hidden layers
        for l in reversed(range(1, n)):
            f_prime = self.activator.deriv_layer(self.a[l])
            self.d[l] = np.multiply(self.w[l].T.dot(self.d[l+1].reshape((len(self.d[l+1]),1))).T, f_prime)[0]

    def compute_partials(self):
        '''compute partial derivativaes of the loss with respect to the weights and biases'''
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

    def pretty_print(self):
        '''prints the general structure of the network'''
        print('Number of input neurons:\t%d' % self.m)
        print('Number of hidden layers:\t%d' % (len(self.ls) - 2))
        for i in range(1, len(self.ls) - 1):
            print('\tNeurons in layer %d:\t%d' % (i, self.ls[i]))
        print('Number of output neurons:\t%d' % self.k)
        print('Activator:\t%s' % str(self.activator).split('(')[0])

    def print_element(self, element, rng, offset=0):
        '''print an element of the network'''
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
