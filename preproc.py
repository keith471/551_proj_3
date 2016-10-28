import numpy as np
import csv

import sklearn
from matplotlib import pyplot as plt
from scipy import sparse
x= np.fromfile('train_x.bin', dtype='uint8')
x= x.reshape((100000,60,60)).astype(float)
x/= 255
x[x<1] = 0
x=x.astype(int)
x=np.lib.pad(x,((0,0),(2,2),(2,2)), 'constant', constant_values=0)

plt.figure(0)
plt.imshow(x[5832],cmap='Greys_r')
plt.show()
print x[0]

