'''Runs our Feed-forward neural net on the mnist data set'''

# need to
# determine best number of layers and neurons per layer
# train and save a model for the best number of layers and neurons
# preprocess the data
# use the saved model to make predictions on the preprocessed data
# submit to kaggle
# http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# http://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde

from __future__ import print_function

from argparse import ArgumentParser
import logging

import sys
import cPickle as pickle
from time import time

from sklearn import metrics

#from matplotlib import pyplot as plt
from skimage.measure import block_reduce

from keras.datasets import mnist
from keras.utils import np_utils

from ffnn import FeedForwardNeuralNet as FFNN
from ffnn import cross_entropy_loss
from postprocess import write_confusion_matrix_to_csv, write_cross_val_results_to_csv, write_to_txt_file
from cross_validation import CrossValidate

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
parser.add_argument('--confusion_matrix',
                    action="store_true",
                    help="save a confusion matrix for the results")
parser.add_argument('--x_val',
                    action="store_true",
                    help="if set, cross-validation will be used")
parser.add_argument('--n_hidden',
                    type=int,
                    help="specify the number of hidden nodes to train the final model with")

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

def show(img):
    plt.figure(0)
    plt.imshow(img, cmap='Greys_r')
    plt.show()

def squish(img, block_size):
    return block_reduce(img, block_size=block_size,func=np.mean)

def get_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # normalize inputs from 0-255 to 0-1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train[X_train > 0.] = 1.0
    #X_train[X_train < 0.5] = 0.0
    X_test[X_test > 0. ] = 1.0
    #X_test[X_test < 0.5 ] = 0.0
    # reshape
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    # one hot encode outputs for training
    y_train = np_utils.to_categorical(y_train)
    return X_train, y_train, X_test, y_test

def get_opt(x_val_results):
    best_err = 1.0
    best_num_neurons = 0
    for num_neurons, train_avgs, test_avgs in x_val_results:
        if test_avgs[1] < best_err:
            best_err = test_avgs[1]
            best_num_neurons = num_neurons
    return best_num_neurons

if __name__ == '__main__':

    ############################################################################
    # load the data
    ############################################################################
    print('loading data...')
    t0 = time()
    if args.squish:
        print('squishing the images by a factor of %d in each dimension' % args.squish)
    all_X_train, all_y_train, X_test, y_test = get_data()
    print('loaded %d training images and %d test images' % (len(all_X_train), len(X_test)))
    print('done in %fs' % (time() - t0))
    print()

    X_train = all_X_train
    y_train = all_y_train

    if args.frac:
        percent = (args.frac * 100.0)
        print("Using only %.f percent of the training data" % percent)
        threshold = int(args.frac * len(X_train))
        if threshold == 0:
            print("Fraction too small, please choose a larger fraction")
            print()
            sys.exit(1)
        X_train = X_train[:threshold]
        y_train = y_train[:threshold]
        print()

    print('shape of all of X_train:')
    print(all_X_train.shape)
    print('shape of X_train:')
    print(X_train.shape)
    print('shape of y_train:')
    print(y_train.shape)
    print('shape of X_test:')
    print(X_test.shape)
    print('shape of y_test:')
    print(y_test.shape)

    '''
    X_train = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    y = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    X_test = [[0, 1], [1, 0], [1,0], [1,0], [0, 1]]
    '''

    ############################################################################
    # CROSS VALIDATION
    ############################################################################
    # cross validate in order to choose best number of nodes per layer
    # we will start with 200 neurons in the hidden layer and add 50 each time

    # define common aspects of the network structure
    # number of inputs (features)
    m = len(X_train[0])
    # number of outputs (classes)
    k = len(y_train[0])

    alpha = 0.01
    lmda = 0.0001
    n_epochs = 20

    if args.x_val:
        # we will want to record the number of hidden neurons, average training error and loss, average testing error and loss
        results = []
        print('cross-validating...')
        for num_hidden_neurons in range(200, 900, 50):
            print('*'*80)
            print('-'*80)
            print('Hidden neurons: %d' % num_hidden_neurons)
            print('_'*80)
            print('*'*80)
            # create a model with this many neurons
            hidden_layer_sizes = [num_hidden_neurons]

            ffnn = FFNN(m, hidden_layer_sizes, k, alpha, lmda, n_epochs, batch_size=args.batch_size, verbose=True)
            ffnn.pretty_print()

            # cross validate the model to get average errs and losses
            cross_validator = CrossValidate(X_train, y_train, ffnn, cv=3)
            t0 = time()
            train_avgs, test_avgs = cross_validator.cross_validate()
            print('done x-validating for %d hidden neurons in %fs' % (num_hidden_neurons, (time() - t0)))
            print()
            results.append((num_hidden_neurons, train_avgs, test_avgs))

        print()
        print('-'*80)
        print('completed cross-validation')
        print('_'*80)
        print()

        ############################################################################
        # save the results for safe keeping
        ############################################################################
        print('writing cross-validation results to csv')
        write_cross_val_results_to_csv(results)
        print('done')
        print()

        ############################################################################
        # determine the best number of nodes and train a model for that number of nodes and save it
        ############################################################################
        opt_hidden_neurons = get_opt(results)
        print('best number of hidden neurons: %d' % opt_hidden_neurons)
        print()

        write_to_txt_file(opt_hidden_neurons, 'opt_hidden_neurons')

    if args.n_hidden:
        opt_hidden_neurons = args.n_hidden
    # we'll want to use more epochs now and all the original data
    n_epochs = 50
    X_train = all_X_train
    y_train = all_y_train

    ffnn = FFNN(m, [opt_hidden_neurons], k, alpha, lmda, n_epochs, batch_size=args.batch_size, verbose=True)
    ffnn.pretty_print()

    print('training the optimal network...')
    t0 = time()
    ffnn.fit(X_train, y_train, use_test_set=True, pickle_best=True, write_errs=True)
    print('done in %fs' % (time() - t0))
    print()

    print('making predictions...')
    t0 = time()
    pred = ffnn.predict(X_test)
    print('done in %fs' % (time() - t0))
    print()

    print('accuracy:')
    print(metrics.accuracy_score(y_test, pred))
    print()

    if args.confusion_matrix:
        print("confusion matrix:")
        conf_matrix = metrics.confusion_matrix(y_test, pred)
        print(conf_matrix)
        # save the confusion matrix to disk
        print('writing confusion matrix to disk')
        write_confusion_matrix_to_csv(conf_matrix)
        print('done')
        print()

    print('DONE!!!!')
    print()
