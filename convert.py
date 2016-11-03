

import cPickle as pickle
from postprocess import write_cross_val_results_to_csv, write_to_txt_file

def read_pickle():
    with open('cross_val_results_1478128087.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    '''
    data = read_pickle()
    write_cross_val_results_to_csv(data)
    '''
    write_to_txt_file(1000, 'testing')
