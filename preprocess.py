'''Preprocessing'''
from __future__ import print_function
import numpy as np
import csv
from matplotlib import pyplot as plt

def one_hot(y):
    y_hot = []
    for s in y:
        v = np.zeros(19, dtype=int)
        v[s] = 1
        y_hot.append(v)
    return y_hot

def default_preprocess(imgs_train, y, imgs_test):
    imgs_train, imgs_test = convert_to_bw(imgs_train, imgs_test)
    y = one_hot(y)
    # convert images to feature vectors
    imgs_train = imgs_train.reshape((imgs_train.shape[0], imgs_train.shape[1]**2,))
    imgs_test = imgs_test.reshape((imgs_test.shape[0], imgs_test.shape[1]**2,))
    return imgs_train, y, imgs_test

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
                data.append(row[1])
        return data

    def read_data(self):
        ''' reads all train and test data and returns as three arrays '''
        imgs_train = self.read_imgs('./data/train_x.bin', 100000)
        y = self.read_csv('./data/train_y.csv')
        imgs_test = self.read_imgs('./data/test_x.bin', 20000)
        return imgs_train, y, imgs_test

    def preprocess(self):
        imgs_train, y, imgs_test = self.read_data()
        return self.process_func(imgs_train, y, imgs_test)

    def show(self, img):
        #5832 shows dotted lines
        plt.figure(0)
        plt.imshow(img, cmap='Greys_r')
        plt.show()
        #print imgs[0]

if __name__ == '__main__':
    pp = Preprocessor()
    imgs_train, y, imgs_test = pp.read_data()
    pp.show(imgs_train[0])
