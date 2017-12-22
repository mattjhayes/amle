# -*- coding: utf-8 -*-
"""
The mlnn5 module provides a multi-layer neural network
algorithm with biases

Based on various public tutorials, including:

* A Step by Step Backpropagation Example by Matt Mazur:
  https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

* How to build a multi-layered neural network in Python by Milo Spencer-Harper:
  https://medium.com/technology-invention-and-more/how-to-build-a-multi \
                            -layered-neural-network-in-python-53ec3d1d326a
"""

import sys

import numpy as np

class Algorithm(object):
    """
    An algorithm module for import by AMLE
    """
    def __init__(self, logger, parameters):
        """
        Initialise the algorithm
        """
        self.logger = logger

        #*** Retrieve parameters passed to us:
        self.seed = parameters['random_seed']

        #*** list with dict per layer specifing number of inputs,
        #*** number of neurons and arbitrary layer name.
        self.layers_config = parameters['layers_config']

        #*** Seed the random number generator:
        if self.seed:
            np.random.seed(self.seed)

        #*** Create NeuralNetwork instance:
        self.neuralnet = NeuralNetwork(logger)

        #*** Add neural layers to network:
        self.neuralnet.add_layers(self.layers_config)

    def initialise(self):
        """
        Use this to re-initialise before re-training
        """
        self.neuralnet.delete_layers()
        #*** Add neural layers to network:
        self.neuralnet.add_layers(self.layers_config)

    def train(self, datasets, parameters):
        """
        Train the multi-layer neural network with training data
        """
        #*** Retrieve parameters passed to us:
        dataset = parameters['dataset']
        partition = parameters['partition']
        iterations = parameters['iterations']

        #*** Get the dataset ready:
        training_dset = datasets[dataset]
        training_inputs = training_dset.inputs_array(partition=partition)
        training_outputs = training_dset.outputs_array(partition=partition)

        self.logger.debug("training_inputs=\n%s", training_inputs)
        self.logger.debug("training_outputs=\n%s", training_outputs)

        # Train the neural network using the training set.
        # Do it many times and make small adjustments each time.
        self.neuralnet.train(training_inputs, training_outputs, iterations)

        self.logger.debug("Trained weights:\n%s", self.neuralnet.weights())

    def test(self, datasets, parameters):
        """
        Ask the multi-layer neural network for an answer to a situation
        and check result accuracy
        """
        #*** Retrieve parameters passed to us:
        dataset = parameters['dataset']
        partition = parameters['partition']

        #*** Get the dataset ready:
        test_dset = datasets[dataset]
        test_inputs = test_dset.inputs_array(partition=partition)
        test_outputs = test_dset.outputs_array(partition=partition)

        self.logger.debug("test_inputs=\n%s", test_inputs)
        self.logger.debug("test_outputs=\n%s", test_outputs)

        #*** Run thinking tests:
        results = []
        for index, input_array in enumerate(test_inputs):
            self.logger.debug("input_array=%s", input_array)
            output = self.neuralnet.think(input_array)
            self.logger.debug("output=%s", output)
            self.logger.debug("correct output=%s", test_outputs[index])
            results.append({'computed': output[0],
                                             'actual': test_outputs[index][0]})
        return results

class NeuralNetwork(object):
    """
    Represents a neural network
    """
    def __init__(self, logger, learning_rate=0.5):
        """
        Initialise the neural network
        """
        self.logger = logger
        # Learning rate for backpropagation weight adjustments:
        self.learning_rate = learning_rate
        #*** Holds classes representing neural layers
        self.layers = []
        #*** Name string for debug etc:
        self.name = ""

    def add_layers(self, layers_config):
        """
        Add neural network layers as per a config. Passed a
        list with dict per layer specifing number of inputs,
        number of neurons and arbitrary layer name.
        List index 0 is first layer (no separate input layer).
        Last configured layer is output layer.
        """
        #*** Reset layers:
        self.layers = []
        #*** Used for sanity check:
        prev_neurons = 0
        #*** Iterate through config adding neural layers:
        for cfg in layers_config:
            self.add_layer(inputs=cfg['inputs'],
                                      neurons=cfg['neurons'], name=cfg['name'])
            #*** Sanity check on number of inputs:
            if prev_neurons and prev_neurons != cfg['inputs']:
                self.logger.critical("Neural network topology error: Trying to"
                               " feed %s inputs into %s neurons in layer=%s. ",
                                      prev_neurons, cfg['inputs'], cfg['name'])
                self.logger.critical("Please fix in project_policy.yaml")
                sys.exit()
            prev_neurons = cfg['neurons']
        #*** Set flags for first and last layers:
        self.layers[0].is_input_layer = True
        self.layers[-1].is_output_layer = True

    def add_layer(self, inputs, neurons, name, bias=1):
        """
        Add a layer (always added as highest layer)
        Last configured layer is assumed to be output layer
        """
        #*** If lower layers exist then set highest one as not output layer:
        if len(self.layers):
            self.layers[-1].is_output_layer = False
        #*** Add layer:
        self.layers.append(NeuralLayer(self.logger, inputs, neurons, name,
                                                                         bias))
        #*** Set this layer as the output layer:
        self.layers[-1].is_output_layer = True

    def delete_layers(self):
        """
        Delete all layers
        """
        self.layers = []

    def train(self, inputs, target_out, iterations):
        """
        Train the neural network, running a number of training
        iterations (epochs) and adjusting synaptic weights each time
        Passed training inputs with corresponding target outputs and
        number of training iterations to carry out
        """
        #*** Sanity check that number of outputs equals output neurons:
        if self.layers[-1].neurons != target_out.shape[1]:
            self.logger.critical("Number of output neurons (%s) not equal to"
                                    " number of outputs (%s)",
                                    self.layers[-1].neurons,
                                    target_out.shape[1])
            self.logger.critical("Please fix error in project_policy.yaml")
            sys.exit()

        #*** Run training iterations (aka epochs):
        for iteration in xrange(iterations):
            #*** Pass the training input set through the neural network:
            layer_outputs = self.feed_forward(inputs)

            #*** Work backward through each layer calculating errors and
            #*** adjusting weights:
            for index, layer in reversed(list(enumerate(self.layers))):
                #*** Account for outputs having additional initial entry
                #***  for inputs:
                out_index = index + 1

                #*** Special case for output layer:
                if layer.is_output_layer:
                    #*** Get the actual output from the feed forward:
                    out = layer_outputs[out_index]

                    # total errors with respect to net_inputs (aka node_delta):
                    # node delta is the partial derivative of total error with
                    # respect to the neuron (node) net error
                    #
                    # node_delta = ∂E/∂net = -(target_out - out) * out(1 - out)
                    #
                    node_delta = -(target_out - out) * out * (1 - out)
                    prev_node_delta = node_delta

                    # Calculate weight adjustments:
                    layer.weight_adjustments = layer_outputs[index].T.dot \
                                                                   (node_delta)
                else:
                    # Sum rows to get total error for this layer (required
                    # for intermediate layers, as error propogates to multiple
                    # neurons in higher (closer to output) layer). Additionally
                    # multiply by synaptic weights of the higher layer:
                    error_wrt_output = prev_node_delta.dot(prev_synaptic_weights.T)

                    # outputs with respect to total net inputs:
                    # out . (1 - out)
                    outputs_wrt_total_net_inputs = layer_outputs[out_index] * \
                                                 (1 - layer_outputs[out_index])

                    # Remove bias as it is not an output we adjust:
                    outputs_wrt_total_net_inputs = bias_remove(outputs_wrt_total_net_inputs)

                    # node_delta:
                    total_errors_wrt_net_inputs = error_wrt_output * \
                                                   outputs_wrt_total_net_inputs
                    prev_node_delta = total_errors_wrt_net_inputs

                    # Add the bias to the outputs from earlier (closer to input) layer:
                    # Calculate weight adjustments for the layer:
                    layer.weight_adjustments = layer_outputs[index].T.dot \
                                                  (total_errors_wrt_net_inputs)

                # Synaptic weights with bias removed for next iteration:
                prev_synaptic_weights = bias_remove(layer.synaptic_weights,
                                                                    axis='row')

            #*** Adjust the layer weights:
            for layer in self.layers:
                # Arbitrary learning rate, using one from example:
                layer.synaptic_weights -= self.learning_rate * layer.weight_adjustments

    def feed_forward(self, inputs):
        """
        Passed inputs and return outputs from each layer
        """
        results = []
        for layer in self.layers:
            # Example inputs (3 inputs, 7 examples)
            #[[ 0.  0.  1.]
            # [ 0.  1.  1.]
            # [ 1.  0.  1.]
            # [ 0.  1.  0.]
            # [ 1.  0.  0.]
            # [ 1.  1.  1.]
            # [ 0.  0.  0.]]
            #
            if layer.is_input_layer:
                #*** append bias as extra input:
                inputs = bias_add(inputs, layer.bias)
                #*** Add inputs to results for use in backprop:
                results.append(inputs)

            outputs = sigmoid(np.dot(inputs, layer.synaptic_weights))

            #*** Add bias column to output if not output layer:
            if not layer.is_output_layer:
                outputs = bias_add(outputs, layer.bias)

            results.append(outputs)
            inputs = outputs
        return results

    def think(self, inputs):
        """
        Passed inputs and return outputs
        """
        for layer in self.layers:
            if layer.is_input_layer:
                inputs = bias_add(inputs, layer.bias)
            #*** Calculate outputs for this layer:
            outputs = sigmoid(np.dot(inputs, layer.synaptic_weights))

            #*** Add bias column to output if not output layer:
            if not layer.is_output_layer:
                outputs = bias_add(outputs, layer.bias)
            inputs = outputs
        return outputs

    def weights(self):
        """
        Return the synaptic weights for each layer
        """
        results = []
        for layer in self.layers:
            results.append(layer.synaptic_weights)
        return results

class NeuralLayer(object):
    """
    Generates an array of random synaptic weights for a
    layer of the neural network
    """
    def __init__(self, logger, inputs, neurons, name, bias):
        #*** Create ndarray object of random floats between -1 and 1
        #*** Dimensions are inputs+1 (include bias) x neurons:
        self.synaptic_weights = 2 * np.random.random((inputs + 1, neurons)) - 1
        logger.debug("%s synaptic_weights=\n%s", name, self.synaptic_weights)
        #*** Increment number of inputs to account for having a bias:
        self.inputs = inputs + 1
        self.neurons = neurons
        self.name = name
        self.bias = bias
        self.is_input_layer = False
        self.is_output_layer = False
        #*** Placeholder to store training weight adjustments:
        self.weight_adjustments = np.array([0])

#======================= Activation Functions & Derivatives ===================
def sigmoid(x):
    """
    The Sigmoid function, which describes an S shaped curve.
    We pass the weighted sum of the inputs through this function to
    normalise them between 0 and 1. Uses exp (e) from numpy.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    The derivative of the Sigmoid function.
    This is the gradient of the Sigmoid curve.
    It indicates how confident we are about the existing weight.
    """
    return x * (1 - x)

#======================= Bias Helper Functions ================================

def bias_add(array_, bias):
    """
    Add bias column to an array.
    Has to handle case where array is 1-D
    Pass it an array (1-D or 2-D) and a bias value to fill
    Returns array with bias added as column on the right
    """
    if array_.ndim == 1:
        #*** append bias as extra input to 1-D array
        bias_array = np.ones((1, bias), dtype=array_.dtype)
        result = np.append(array_, bias_array)
    else:
        #*** append bias as extra input column to 2-D array:
        bias_array = np.ones((array_.shape[0], bias), dtype=array_.dtype)
        result = np.append(array_, bias_array, axis=1)
    return result

def bias_remove(array_, axis='column'):
    """
    Remove bias column or row from an array.
    Pass it an array (1-D or 2-D) and a bias value to fill
    Returns array with bias removed (either as column on the right
    or row on bottom)
    """
    if axis == 'column':
        result = np.delete(array_, array_.shape[1]-1, axis=1)
    elif axis == 'row':
        result = np.delete(array_, array_.shape[0]-1, axis=0)
    else:
        sys.exit("Error in axis value passed to remove_bias. Exiting")
    return result

#======================= Loss Functions =======================================

def loss_mse(target, actual):
    """
    Mean squared error (loss) function: 1/2 diff ^2
    Pass it numpy arrays and it returns one back
    """
    diff = target - actual
    diff_squared = np.square(diff)
    return 0.5 * diff_squared

def loss_simple(target, actual):
    """
    Simple error (loss) function that just computes difference.
    Pass it numpy arrays and it returns one back
    """
    return target - actual
