'''Trains a convolutional neural net on the mnist dataset using Keras'''
from __future__ import print_function
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import sys
import os.path

from argparse import ArgumentParser

# bug workaround
import tensorflow as tf
tf.python.control_flow_ops = tf

K.set_image_dim_ordering('th')

# Parse options and arguments
parser = ArgumentParser()
parser.add_argument('model_name', type=str,
                    help='the name of the model to save')

args = parser.parse_args()

weights_file_name = args.model_name + '_weights.h5'
model_json_file_name = args.model_name + '_model.json'
if os.path.exists(weights_file_name) or os.path.exists(model_json_file_name):
    parser.error('A file with the name %s or %s already exists' % (weights_file_name, model_json_file_name))
    sys.exit(1)

print(__doc__)
print()
parser.print_help()
print()

def get_model(num_classes):
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def get_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test

def save_model(model, name='model'):
    model_name = name + '_model.json'
    weights_name = name + '_weights.h5'
    print('saving model json to %s and model weights to %s' % (model_name, weights_name))
    model.save_weights(weights_name)
    with open(model_name, 'wb') as f:
        f.write(model.to_json())

def load_model(name='model'):
    model_name = name + '_model.json'
    weights_name = name + '_weights.h5'
    with open(model_name, 'rb') as f:
        json_data = f.read()
    model = model_from_json(json_data)
    model.load_weights(weights_name)
    return model

if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    X_train, y_train, X_test, y_test = get_data()

    num_classes = y_test.shape[1]

    # build the model
    model = get_model(num_classes)
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=9, batch_size=200, verbose=2)

    save_model(model, args.model_name)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
