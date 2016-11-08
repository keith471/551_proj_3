
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
from keras.layers.convolutional import AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from skimage.measure import block_reduce
import csv
import sys
import os.path



# bug workaround


K.set_image_dim_ordering('th')



def get_model(num_classes):
	# create model
	model = Sequential()
	model.add(Convolution2D(128, 11, 11, border_mode='valid', input_shape=(1, 64, 64), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 8, 8, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 5, 5, activation='relu'))

	
	model.add(Dropout(0.50))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	


def get_data():
    # load data
    X= numpy.fromfile('train_x.bin', dtype='uint8')
    X= X.reshape((100000,60,60)).astype(float)

    #normalize and remove grey background
    X/=255
    X[X<1]=0
    X=numpy.lib.pad(X,((0,0),(2,2),(2,2)), 'constant', constant_values=0)

    #load training data
    y=[]
    fh = open('train_y.csv')

    i = 1

    for line in fh:

        label = line.split(',',)[1]

        y.append(label)

        i += 1
    y.pop(0)
    y=numpy.array(y)
    y=y.astype(int)

    # validation split
    X_train, X_test = X[:80000], X[80000:]
    y_train, y_test = y[:80000], y[80000:]


    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 64, 64)
    X_test = X_test.reshape(X_test.shape[0], 1, 64, 64)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test

def save_model(model, name='model'): #function to save model to disk
    model_name = name + '_model.json'
    weights_name = name + '_weights.h5'
    print('saving model json to %s and model weights to %s' % (model_name, weights_name))
    model.save_weights(weights_name)
    with open(model_name, 'wb') as f:
        f.write(model.to_json())

def load_model(name='model'): #load existing model from disk
    model_name = name + '_model.json'
    weights_name = name + '_weights.h5'
    with open(model_name, 'rb') as f:
        json_data = f.read()
    model = model_from_json(json_data)
    model.load_weights(weights_name)
    return model

def writeresult(Y): #create prediction result file
    with open('crazy02.csv', 'wb') as csvfile:
        a = csv.writer(csvfile)
        a.writerow(['Id', 'Prediction'])
        j = 0

        for i in Y:
            a.writerow((j,int(i)))
            j += 1

if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    X_train, y_train, X_test, y_test = get_data()

    #testx = numpy.fromfile('test_x.bin', dtype='uint8')
    #testx = testx.reshape((20000,60,60))

    #testx=testx.astype(float)
    #testx=testx/255
    #testx[testx<1] = 0
    #testx=block_reduce(testx, block_size=(1,2,2),func=numpy.mean)
    #testx=numpy.lib.pad(testx,((0,0),(2,2),(2,2)), 'constant', constant_values=0)
    #testx = testx.reshape(testx.shape[0], 1, 64, 64)


    #model=load_model('name')
    #Y=model.predict_classes(testx)
    #print (Y[0])
    #writeresult(Y)
    num_classes = y_test.shape[1]

    # build the model
    model = get_model(num_classes)
    #model=load_model('name')
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    while(True): #allows training to be stopped at any time while keeping model saved
        model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=200, verbose=2)

        save_model(model, 'nameofmodel01')

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

