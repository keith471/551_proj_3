'''Get some results for alpha'''
from __future__ import print_function

from argparse import ArgumentParser
import logging

import sys
import os.path
import cPickle as pickle
from time import time

from sklearn.model_selection import train_test_split

from ffnn import FeedForwardNeuralNet as FFNN
from ffnn import cross_entropy_loss, squared_error_loss
from preprocess import Preprocessor
from postprocess import write_results

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

def try_alpha(alpha, X_train, X_test, y_train, y_test):
    m = len(X_train[0])
    hidden_layer_sizes = [1800]
    k = len(y_train[0])

    lmda = 0.
    n_epochs = 10

    # instantiate the neural net
    ffnn = FFNN(m, hidden_layer_sizes, k, alpha, lmda, n_epochs, batch_size=args.batch_size, verbose=True)
    ffnn.pretty_print()

    t0 = time()
    ffnn.fit(X_train, y_train)

    return ffnn.get_performance(X_test, y_test)

if __name__ == '__main__':

    # load the data
    print('loading data...')
    t0 = time()
    pp = Preprocessor()
    if args.squish:
        print('squishing the images by a factor of %d in each dimension' % args.squish)
        X_train, y, X_test = pp.preprocess((1, args.squish, args.squish))
    else:
        X_train, y, X_test = pp.preprocess()
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

    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.25)

    print('shape of X_train:')
    print(X_train.shape)
    print('shape of X_test:')
    print(X_test.shape)
    print('shape of y_train:')
    print(y_train.shape)

    # try out various alphas
    l_and_e = []
    alphas = [10**x for x in range(-8, 2)]
    for alpha in alphas:
        loss, error = try_alpha(alpha, X_train, X_test, y_train, y_test)
        l_and_e.append((alpha, loss, error))

    print()
    print('alpha\tloss\terror')
    for alpha, loss, error in l_and_e:
        print('%f\t%f\t%f' % (alpha, loss, error))
