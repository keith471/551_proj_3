# README
In order for the following to work, please ensure that you have the data for this project in a folder called *data* in the same directory as the files you wish to execute.

## Logistic regression

## Feedforward neural network
You can find our feedforward neural network implementation in ffnn.py. To run it, you have two options:
1. Run with `python run.py [options]` to run it on the **given data** (images with two digits). Use `python run.py --help` to get help with the options.
2. Run with `python run_2.py [options]` to run it on the **MNIST data**. Again, use the `--help` flag to see the options.

## Convolutional neural net (using libraries)

The CNN can be ran directly as long as all libraries are installed (numpy, theano or tensorflow, keras, skimage)
Name under which to save CNN model can be changed in line 151
Most recent model is automatically saved, so the process can be stopped at any time.
Model parameters (lines 33 to 45) may be changed at will.
Commented out code in lines 125 to 145 may be ignored, it was used to produce prediction result csv files.
