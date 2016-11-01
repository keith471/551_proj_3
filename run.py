'''Runs our Feed-forward neural net'''
from __future__ import print_function

from argparse import ArgumentParser
import logging

import sys
import os.path

from ffnn import FeedForwardNeuralNet as FFNN
from ffnn import cross_entropy_loss, squared_error_loss
from preprocess import Preprocessor

from time import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

'''
# Parse options and arguments
parser = ArgumentParser()
parser.add_argument('data_folder', type=str,
                    help='the name of the folder containing the text files')
parser.add_argument('--verbose', action='store_true',
                    help='if set, output will be more verbose')
parser.add_argument("--chi2_select",
                    action="store", type=int, dest="select_chi2",
                    help="select some number of features using a chi-squared test")
parser.add_argument('--test',
                    action='store', type=int, dest='test_fraction',
                    help='if set, only a fraction of the data will be trained on and no cross-validation will be used')
parser.add_argument("--dev",
                    action="store_true",
                    help="if set, accuracy will be measured against a 30 percent dev set. Cannot be used in tandem with --cv_range.")
parser.add_argument("--predict",
                    action="store_true",
                    help="If set, predictions will be made for the unknown test data")

args = parser.parse_args()

if not os.path.exists(args.data_folder):
    parser.error('The folder %s does not exist' % args.data_folder)
    sys.exit(1)

print(__doc__)
print()
parser.print_help()
print()

'''

if __name__ == '__main__':

    # load the data
    print('loading data...')
    t0 = time()
    pp = Preprocessor()
    X_train, y, X_test = pp.preprocess()
    print('loaded %d training images and %d test images' % (len(X_train), len(X_test)))
    print('shape of X_train:')
    print(X_train.shape)
    print('shape of X_test:')
    print(X_test.shape)
    print('shape of y:')
    print(y.shape)
    print('done in %fs' % (time() - t0))
    print()

    print('initializing a network...')
    t0 = time()
    # define the network structure
    # number of inputs (features)
    m = len(X_train[0])
    # sizes of hidden layers
    hidden_layer_sizes = [1800]
    # number of outputs (classes)
    k = len(y[0])

    # define alpha (initial learning rate) and lambda (regularization penalty)
    alpha = 1e-1
    lmda = 1e-3

    # instantiate the neural net
    ffnn = FFNN(m, hidden_layer_sizes, k, alpha, lmda, verbose=True)
    ffnn.pretty_print()
    print('done in %fs' % (time() - t0))
    print()

    sys.exit(1)

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
