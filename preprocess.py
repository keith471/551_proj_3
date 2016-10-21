import numpy
import scipy.misc # to visualize only
x = numpy.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))
scipy.misc.imshow(x[0]) # to visualize only
