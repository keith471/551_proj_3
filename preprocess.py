'''Preprocessing
This script is executable. If you execute it, it will save preprocessed values of X_train, y, and X_test'''
from __future__ import print_function
import numpy as np
import csv
from skimage.measure import block_reduce
#from matplotlib import pyplot as plt
import sys
'''
import cPickle as pickle
import os.path
from argparse import ArgumentParser

# Parse options and arguments
parser = ArgumentParser()
parser.add_argument('data_name', type=str,
                    help='a name that will be appended to the filenames for the data when saved')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='sef if you would like to overwrite existing files')

args = parser.parse_args()

if os.path.exists('preprocessed_data/' + args.data_name) and not args.overwrite:
    parser.error('Files corresponding to that name already exist. Use --overwrite to overwrite them')
    sys.exit(1)

print(__doc__)
print()
parser.print_help()
print()
'''

# PREPROCESSING HELPERS

def one_hot(y):
    y_hot = []
    for s in y:
        v = np.zeros(19, dtype=int)
        v[s] = 1
        y_hot.append(v)
    return np.array(y_hot)

def convert_to_bw(imgs_train, imgs_test):
    '''remove backgroud pattern'''
    imgs_train /= 255
    imgs_train[imgs_train < 1.0] = 0.0
    imgs_train = imgs_train.astype(int)
    imgs_test /= 255
    imgs_test[imgs_test < 1.0] = 0.0
    imgs_test = imgs_test.astype(int)
    return imgs_train, imgs_test

def pad(imgs):
    return np.lib.pad(imgs, ((0,0), (2,2), (2,2)), 'constant', constant_values=0)

def squish(imgs, block_size):
    return block_reduce(imgs, block_size=block_size,func=np.mean)

# FOR DEBUGGING AND VISUALIZING

def show(img):
    #5832 shows dotted lines
    plt.figure(0)
    plt.imshow(img, cmap='Greys_r')
    plt.show()

# PREPROCESSING TECHNIQUES

def default_preprocess(imgs_train, y, imgs_test, block_size):
    '''returns data in desired X_train, y, X_test format'''
    imgs_train, imgs_test = convert_to_bw(imgs_train, imgs_test)
    if block_size:
        print('squishing')
        imgs_train = squish(imgs_train, block_size)
        imgs_test = squish(imgs_test, block_size)
        #imgs_train[imgs_train > 0.0] = 1.0
        #imgs_test[imgs_test > 0.0] = 1.0
    y = one_hot(y)
    # convert images to feature vectors
    imgs_train = imgs_train.reshape((imgs_train.shape[0], imgs_train.shape[1]**2,))
    imgs_test = imgs_test.reshape((imgs_test.shape[0], imgs_test.shape[1]**2,))
    return imgs_train, y, imgs_test

# PREPROCESSOR

class Preprocessor(object):

    def __init__(self, process_func=default_preprocess):
        self.process_func = process_func

    def read_imgs(self, fpath, num):
        imgs = np.fromfile(fpath, dtype='uint8')
        imgs = imgs.reshape((num, 60, 60)).astype(float)
        return imgs

    def read_csv(self, fpath):
        ''' returns an array containing the data from the file'''
        data = []
        with open(fpath, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip the header
            reader.next()
            for row in reader:
                # row[0] contains the id, row[1] contains the class
                data.append(int(row[1]))
        return data

    def read_data(self):
        ''' reads all train and test data and returns as three arrays '''
        imgs_train = self.read_imgs('./data/train_x.bin', 100000)
        y = self.read_csv('./data/train_y.csv')
        imgs_test = self.read_imgs('./data/test_x.bin', 20000)
        return imgs_train, y, imgs_test

    def preprocess(self, block_size=None):
        imgs_train, y, imgs_test = self.read_data()
        return self.process_func(imgs_train, y, imgs_test, block_size)

    def load_pickle(self, name):
        name = 'preprocessed_data/' + name
        with open(name, 'rb') as f:
            p = pickle.load(f)
        return p

    def load(self, name):
        '''load X_train, y, and X_test tensors that have already been pickled'''
        X_train_name = name + '_X_train.pkl'
        y_name = 'one_hot_y.pkl'
        X_test_name = name + '_X_test.pkl'
        X_train = self.load_pickle(X_train_name)
        y = self.load_pickle(y_name)
        X_test = self.load_pickle(X_test_name)
        return X_train, y, X_test

    def save_pickle(self, name, content):
        name = 'preprocessed_data/' + name
        with open(name, 'wb') as f:
            pickle.dump(content, f)

    def save(self, name):
        '''save X_train, y, and X_test tensors in pickle files'''
        X_train, y, X_test = self.preprocess()
        X_train_name = name + '_X_train.pkl'
        y_name = 'one_hot_y.pkl'
        X_test_name = name + '_X_test.pkl'
        self.save_pickle(X_train_name, X_train)
        self.save_pickle(y_name, y)
        self.save_pickle(X_test_name, X_test)
        print('saved X_train, y, and X_test as %s, %s, and %s' % (X_train_name, y_name, X_test_name))

'''
if __name__ == '__main__':

    pp = Preprocessor()
    pp.preprocess((1,2,2))
'''
