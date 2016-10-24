'''Feed-forward neural-net'''

# TODO
# add InputLayer class
# replace NNLayer with HiddenLayer
# Just keep it simple and treat it as a single layer neural net

import numpy as np
import math

class FeedForwardNeuralNet(object):
    def __init__(self):
        # TODO
        pass

    def fprop(self, x):
        # TODO
        # somehow feed x into the input layer
        # for each Layer
        #   compute the output of all neurons in the Layer
        #   and set this output as the input to the next Layer
        # finally, compute the output of the last Layer and return this
        for layer in self.network.layers:
            for neuron in layer.neurons:
                neuron.fire()
        return self.network.get_output()

    def bprop(self):
        # TODO
        # assume this is called with the correction of the output nodes already set
        # for each hidden unit, compute its share of the correction
        # update each network weight

        # set delta on the output
        self.network.output_layer.set_delta()

        for layer in reversed(self.network.hidden_layers):
            for neuron in layer.neurons:

                neuron.update_out_weights()

    def initialize(self):
        '''initialize the network'''
        # basically, just need to set up the neurons and initialize the weights
        # set self.network to...
        pass
        # TODO

################################################################################
# activation functions
################################################################################
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

################################################################################
# network components
################################################################################

# nodes

class GraphNode(object):
    def __init__(self, in_edges, out_edges):
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.parents = self.get_parents()
        self.children = self.get_children()

    def get_parents(self):
        return [e.tail for e in self.in_edges]

    def get_children(self):
        return [e.head for e in self.out_edges]

class Neuron(GraphNode):
    def __init__(self, in_edges, out_edges, output_activation_func=sigmoid):
        GraphNode.__init__(self, in_edges, out_edges)
        self.w, self.x = self.get_w_and_x(in_edges)
        self.output_activation_func = output_activation_func

    def get_w_and_x(self, in_edges):
        # weights are 1 for InputNeurons
        w_l = [e.weight for e in in_edges]
        x_l = [e.value for e in in_edges]
        return np.array(w_l), np.array(x_l)

    def output_activation(self):
        '''output value of the neuron'''
        return self.output_activation_func(self.input_activation())

    def fire(self):
        '''fire the synapse between this neuron and its children; i.e. push a value
        on its outgoing edges'''
        output = self.output_activation()
        for edge in self.out_edges:
            edge.set_value(output)

    def update_out_weights(self):
        '''a neuron should be able to update the weights of its output edges'''
        # TODO
        for edge in self.out_edges:
            # calculate delta
                edge.update_weight()

class InputNeuron(Neuron):
    def __init__(self, in_edges, out_edges, output_activation_func=sigmoid):
        Neuron.__init__(self, in_edges, out_edges, output_activation_func)

    # ????? Does input node take a bias term?
    def input_activation(self):

        return np.dot(self.w, self.x)

    def get_w_and_x(self, in_edges):
        # override with 1 for weights
        w_l = [1.0 for e in in_edges]
        x_l = [e.value for e in in_edges]
        return np.array(w_l), np.array(x_l)

    def get_parents(self):
        # override
        return None


class HiddenNeuron(Neuron):

    def __init__(self, bias_edge, in_edges, out_edges, output_activation_func=sigmoid):
        '''takes the bias, sets of input and output edges, as well as the output activation function'''
        Neuron.__init__(self, in_edges, out_edges, output_activation_func)
        # remove this if all nodes take a bias
        self.bias_edge = bias_edge

    def input_activation(self):
        return self.bias_edge.weight + np.dot(self.w, self.x)

    # unique to hidden and output neurons
    def set_delta(self):
        output = self.output_activation()
        self.delta = output * (1 - output) *

class OutputNeuron(Neuron):

    def __init__(self, bias_edge, in_edges, out_edges, output_activation_func):
        Neuron.__init__(self, in_edges, out_edges, output_activation_func)
        # remove this if all nodes take a bias
        self.bias_edge = bias_edge

    def input_activation(self):
        return self.bias_edge.weight + np.dot(self.w, self.x)

    def set_delta(self):
        output = self.get_output()
        # what is y?
        self.delta = output * (1 - output) * (y - output)

# edges

class GraphEdge(object):
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.value = value

class Synapse(GraphEdge):
    def __init__(self, head, tail, weight, value):
        GraphEdge.__init__(self, head, tail, weight)
        self.value = value

class BiasEdge(Synapse):
    def __init__(self, head, tail, weight):
        Synapse.__init__(self, head, tail, weight, None)

class InputEdge(Synapse):
    def __init__(self, head, tail, value):
        Synapse.__init__(self, head, tail, 1.0, value)

    def set_value(self, value):
        '''set with the output of self.tail'''
        self.value = value

    def update_weight(self):
        # get delta

class HiddenEdge(Synapse):
    def __init__(self, head, tail, weight, value):
        Synapse.__init__(self, head, tail, weight, value)

class OutputEdge(Synapse):
    def __init__(self, head, value):
        Synapse.__init__(self, head, None, None, value)

# layer

class GraphLayer(object):
    def __init__(self, neurons, prev, next):
        self.neurons = neurons
        self.prev = prev
        self.next = next

class NNLayer(GraphLayer):
    def __init__(self, neurons, prev, next):
        GraphLayer.__init__(self, neurons, prev, next)

class OutputLayer(GraphLayer):
    def __init__(self, neurons, prev):
        GraphLayer.__init__(self, neurons, prev, None)

    def get_output(self):
        return self.neurons[0].get_output()

    def set_deltas(self):
        for neuron in self.neurons:
            neuron.set_delta()
# network

def smash(ll):
    return [item for sublist in ll for item in sublist]

class NNNetwork(object):

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        layers = [input_layer, hidden_layers, output_layer]
        self.layers = smash(layers)

    def get_output(self):
        '''get the output of the network'''
        return self.output_layer.get_output()

    def get_output_correction(self):
        return self.output_layer.get_correction()
