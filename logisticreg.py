from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import numpy as np

#load and preprocess data
x= np.fromfile('train_x.bin', dtype='uint8')
x= x.reshape((100000,60,60)).astype(float)
x/= 255
x[x<1] = 0

Y=[]
fh = open('train_y.csv')

i = 1

for line in fh:

    label = line.split(',',)[1]

    Y.append(label)

    i += 1
Y.pop(0)

Ytest=[]

# use sklearn's logistic regression
clf = LogisticRegression()

x = x.reshape(x.shape[0], 60*60)
score = cross_val_score(clf, x, Y, cv=5) #cross validation

print("logistic regression score: %0.8f " % (score))
