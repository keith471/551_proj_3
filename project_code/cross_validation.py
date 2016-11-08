'''Cross validation'''
from __future__ import print_function

from time import time

from data_partitioner import DataPartitioner

class CrossValidate:

    def __init__(self, X, y, clf, cv=10):
        self.X = X
        self.y = y
        self.cv = cv
        self.clf = clf
        self.partitioner = DataPartitioner(cv, X, y)

    def get_avgs(self, l_and_e):
        zipped = zip(*l_and_e)
        avg_loss = sum(zipped[0]) / float(len(zipped[0]))
        avg_err = sum(zipped[1]) / float(len(zipped[1]))
        return (avg_loss, avg_err)

    def cross_validate(self):
        '''Trains and tests the given classifier on cv folds, and returns the average loss and error
        against the validation set'''
        train_loss_and_err = []
        test_loss_and_err = []
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.partitioner.getPartitions()):
            print('-' * 60)
            print("Training on training set %d" % i)
            print('_' * 60)
            t0 = time()
            self.clf.fit(X_train, y_train)
            dur = time() - t0
            print("completed training in %fs" % dur)
            print()
            print("measuring performance against training and test sets")
            t0 = time()
            train_loss, train_err = self.clf.get_performance(X_train, y_train)
            test_loss, test_err = self.clf.get_performance(X_test, y_test)
            train_loss_and_err.append((train_loss, train_err))
            test_loss_and_err.append((test_loss, test_err))
            dur = time() - t0
            print("completed performance measurements in %fs" % dur)
            print()
            print("Train loss and error of %dth partition:" % i)
            print('%f\t%f' % (train_loss, train_err))
            print("Test loss and error of %dth partition:" % i)
            print('%f\t%f' % (test_loss, test_err))
            print()
        train_avgs = self.get_avgs(train_loss_and_err)
        test_avgs = self.get_avgs(test_loss_and_err)
        print("Average training loss and error:")
        print('%f\t%f' % train_avgs)
        print("Average testing loss and error:")
        print('%f\t%f' % test_avgs)
        print()
        return train_avgs, test_avgs
