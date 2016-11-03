'''Postprocessing'''
from __future__ import print_function

import csv
import sys
from time import time

import cPickle as pickle

def write_to_csv(filename, data, fieldnames):
    ''' writes and array of data to csv with field names fieldNames'''
    # append time to filename to ensure uniqueness
    uniqueFilename = filename + "_%.f" % time()
    print("Writing predictions for %s classifier to %s.csv" % (filename, uniqueFilename))
    print()
    with open(uniqueFilename + '.csv', 'w') as mycsvfile:
        writer = csv.writer(mycsvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

def write_results(pred, filename='predictions'):
    ''' takes an array of predictions and converts it into an array of id, prediction entries '''
    data = [[i,v] for i,v in enumerate(pred)]
    fieldnames = ['id', 'category']
    write_to_csv(filename, data, fieldnames)

def post_process(results):
    for i, v in results:
        filename = "%s_%d" % (v[0], i)
        write_results(filename, v[1])

def to_pickle(data, name):
    name = name + '_%.f.pkl' % time()
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def write_errs_to_csv(train, dev):
    '''writes training/development loss/error to csv'''
    fieldnames = ['iter', 'train_loss', 'train_err', 'dev_loss', 'dev_err']
    # reformat data
    data = []
    for i in range(len(train)):
        row = []
        row.append(i+1)
        row.append(train[i][0])
        row.append(train[i][1])
        row.append(dev[i][0])
        row.append(dev[i][1])
        data.append(row)
    unique_fname = 'errs_%.f.csv' % time()
    with open(unique_fname, 'w') as mycsvfile:
        writer = csv.writer(mycsvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

def write_confusion_matrix_to_csv(m):
    unique_fname = 'confusion_matrix_%.f.csv' % time()
    with open(unique_fname, 'wb') as mycsvfile:
        writer = csv.writer(mycsvfile)
        writer.writerows(m)

def write_cross_val_results_to_csv(results):
    unique_fname = 'cross_val_results_%.f.csv' % time()
    fieldnames = ['num_hidden_neurons', 'avg_train_loss', 'avg_train_err', 'avg_test_loss', 'avg_test_err']
    data = []
    for num_hidden_neurons, train_avgs, test_avgs in results:
        row = []
        row.append(num_hidden_neurons)
        row.append(train_avgs[0])
        row.append(train_avgs[1])
        row.append(test_avgs[0])
        row.append(test_avgs[1])
        data.append(row)
    with open(unique_fname, 'wb') as mycsvfile:
        writer = csv.writer(mycsvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

def write_to_txt_file(value, name):
    unique_fname = name + '_%.f.txt' % time()
    if type(value) != str:
        value = str(value)
    with open(unique_fname, 'w') as f:
        f.write(value)
