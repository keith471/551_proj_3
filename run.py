'''Runs our Feed-forward neural net'''
from __future__ import print_function

from argparse import ArgumentParser
import logging

import sys
import os.path

from ffnn import FeedForwardNeuralNet as FFNN
from ffnn import cross_entropy_loss, squared_error_loss
from preprocess import Preprocessor

import cPickle as pickle

from time import time

#from keras.datasets import mnist
#from keras.utils import np_utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Parse options and arguments
parser = ArgumentParser()
parser.add_argument("--batch_size",
                    action="store", type=int, default=200,
                    help="the batch size to use between weight and bias updates")
parser.add_argument('--frac', type=float,
                    help='if set, only a fraction of the training data will be used; valid values are in the range (0, 1]')
parser.add_argument('--squish', type=int,
                    help='if set, each dimension of the image will be squished by the number specified, e.g. if --squish=2 then images will be made 1/4 their original size')

args = parser.parse_args()

if args.batch_size < 1:
    parser.error('The batch size must be positive')
    sys.exit(1)

if args.squish:
    if args.squish < 1:
        parser.error('The squish ratio must be positive')
        sys.exit(1)

if args.frac and (args.frac <= 0.0 or args.frac > 1.0):
    parser.error('The fraction must be in the range (0,1]')
    sys.exit(1)

print(__doc__)
print()
parser.print_help()
print()

def get_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':

    # load the data
    print('loading data...')
    t0 = time()
    pp = Preprocessor()
    if args.squish:
        print('squishing the images by a factor of %d in each dimension' % args.squish)
    X_train, y, X_test = pp.preprocess((1, args.squish, args.squish))
    print('loaded %d training images and %d test images' % (len(X_train), len(X_test)))
    print('done in %fs' % (time() - t0))
    print()

    if args.frac:
        percent = (args.frac * 100.0)
        print("Using only %.f percent of the training data" % percent)
        threshold = int(args.frac * len(X_train))
        if threshold == 0:
            print("Fraction too small, please choose a larger fraction")
            print()
            sys.exit(1)
        X_train = X_train[:threshold]
        y = y[:threshold]
        print()

    print('shape of X_train:')
    print(X_train.shape)
    print('shape of X_test:')
    print(X_test.shape)
    print('shape of y:')
    print(y.shape)

    '''
    X_train = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    y = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    X_test = [[0, 1], [1, 0], [1,0], [1,0], [0, 1]]
    '''

    '''
    X_train, y, X_test, y_test = get_data()
    '''

    print('initializing a network...')
    t0 = time()
    # define the network structure
    # number of inputs (features)
    m = len(X_train[0])
    # sizes of hidden layers
    hidden_layer_sizes = [500]
    # number of outputs (classes)
    k = len(y[0])

    # define alpha (initial learning rate) and lambda (regularization penalty)
    alpha = 0.01
    lmda = 0.0001
    n_epochs = 500

    # instantiate the neural net
    ffnn = FFNN(m, hidden_layer_sizes, k, alpha, lmda, n_epochs, batch_size=args.batch_size, verbose=True)
    ffnn.pretty_print()
    print('done in %fs' % (time() - t0))
    print()

    print('training the network...')
    t0 = time()
    ffnn.fit(X_train, y)
    print('done in %fs' % (time() - t0))
    print()

    print('making predictions...')
    t0 = time()
    pred = ffnn.predict(X_test)
    print('done in %fs' % (time() - t0))
    print()

    print(pred)
