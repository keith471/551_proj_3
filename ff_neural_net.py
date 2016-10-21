
# Basic idea

# Select a training example
# Run it through the model to get its output using fprop
# Update the weights using gradient descent and bprop
# Repeat for all training examples
# Then perhaps run model on say a 20% validation set to get validation error and
# rerun if need be?

# GRADIENT DESCENT TIPS
# Stochastic gradient descent:
#   compute error on ONE training example at a time and use results to update all edge weights
# Decay learning rate!
#   start with a large learning rate (0.1)
#   when validation error stops improving, divide learning rate by 2 and repeat until division of learning rate further results in no
#   further improvement in validation error
# Try running on a minibatch rather that one sample at a time for greater speed
#   The gradient is the average regularized loss for the mini-batch
# run first on a small dataset to debug (see slides 76-80)
# to make sure you are calculating the gradient properly, compare your gradient with a finite difference approx of the gradient
#   df(x)/dx ~= (f(x+e) - f(x-e))/2e where e is something small

################################################################################
# initialization
################################################################################
# choose some arbitrary number of nodes in the hidden layer
#   finalize a model for this number of nodes and measure its performance
#   (later repeat for other numbers of nodes and choose the number that gives the best performance)
# initialization of weights and biases - see slide 70

################################################################################
# fprop
################################################################################
# implement forward propogation as an acyclic flow graph
# each node is an object with an fprop method that computes the value of the box given its children
# Then, you just need to call the fprop method of each node in the right order
#   ==> definitely could be recursion here

################################################################################
# bprop
################################################################################
# each node also has a bprop method
#   computes the gradient of the loss with respect to each of its children
#   fprop depends on the fprop of the node's children while
#   bprop depends on the brop of the node's parents

################################################################################
# final layer
################################################################################
# we have 19 possible classes (0-18) (or should we have a class per possible combo of numbers?)
#   use softmax at the last stage

################################################################################
# tuning hyperparameters
################################################################################
# use cross validation to select
#   hidden layer size
#   learning rate
#   number of iterations/epochs
# Use grid or random search to try to find best model parameters


##
