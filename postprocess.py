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
