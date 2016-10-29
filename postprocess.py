'''Postprocessing'''
from __future__ import print_function

import csv
import sys
from time import time

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
