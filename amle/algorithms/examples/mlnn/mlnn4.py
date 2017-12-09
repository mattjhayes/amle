"""
The mlnn4 module provides a multi-layer neural network
algorithm with biases

Based on:
* Tutorial 'How to build a multi-layered neural network in Python'
  by Milo Spencer-Harper
  See: https://medium.com/technology-invention-and-more/how-to-build-a-multi \
                            -layered-neural-network-in-python-53ec3d1d326a
"""

import sys

import copy

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
    def __init__(self, logger):
        """
        Initialise the neural network
        """
        self.logger = logger
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

    def train(self, inputs, outputs, iterations):
        """
        Train the neural network, running a number of training
        iterations (epochs) and adjusting synaptic weights each time
        """
        #*** Sanity check that number of outputs equals output neurons:
        if self.layers[-1].neurons != outputs.shape[1]:
            self.logger.critical("Number of output neurons (%s) not equal to"
                                    " number of outputs (%s)",
                                    self.layers[-1].neurons,
                                    outputs.shape[1])
            self.logger.critical("Please fix error in project_policy.yaml")
            sys.exit()

        #*** Run training iterations (aka epochs):
        for iteration in xrange(iterations):
            #*** Pass the training input set through the neural network:
            layer_outputs = self.feed_forward(inputs)

            #*** Work backward through each layer calculating errors and
            #*** adjusting weights:
            for index, layer in reversed(list(enumerate(self.layers))):
                if layer.is_output_layer:
                    #*** Special case for output layer:
                    layer_error = outputs - layer_outputs[index + 1]
                else:
                    layer_error = prev_layer_error_squashed.dot(prev_synaptic_weights.T)

                #*** Calculate the gradient for error correction:
                if layer.is_output_layer:
                    #*** No bias to remove as is output layer:
                    layer_output = layer_outputs[index + 1]
                else:
                    #*** Remove bias from outputs:
                    layer_output = remove_bias(layer_outputs[index + 1], axis='column')

                #*** Squash the output and multiply by the error:
                layer_error_squashed = layer_error * sigmoid_derivative(layer_output)

                #*** Calculate weight adjustment for the layer:
                if not layer.is_input_layer:
                    layer_adjustment = layer_outputs[index].T.dot(layer_error_squashed)
                else:
                    #*** Special case for first layer, use inputs:
                    layer_adjustment = layer_outputs[0].T.dot(layer_error_squashed)

                #*** Copies for use in next iteration (bias removed from synaptic weights):
                prev_synaptic_weights = remove_bias(layer.synaptic_weights, axis='row')
                prev_layer_error_squashed = copy.copy(layer_error_squashed)

                #*** Adjust the layer weights:
                layer.synaptic_weights += layer_adjustment

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
                inputs = add_bias(inputs, layer.bias)
                first_layer = False
                #*** Add inputs to results for use in backprop:
                results.append(inputs)

            outputs = sigmoid(np.dot(inputs, layer.synaptic_weights))

            #*** Add bias column to output if not output layer:
            if not layer.is_output_layer:
                outputs = add_bias(outputs, layer.bias)

            results.append(outputs)
            inputs = outputs
        return results

    def think(self, inputs):
        """
        Passed inputs and return outputs
        """
        for layer in self.layers:
            if layer.is_input_layer:
                inputs = add_bias(inputs, layer.bias)
            #*** Calculate outputs for this layer:
            outputs = sigmoid(np.dot(inputs, layer.synaptic_weights))

            #*** Add bias column to output if not output layer:
            if not layer.is_output_layer:
                outputs = add_bias(outputs, layer.bias)
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

#=========================== Supporting Functions =============================
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

def add_bias(array_, bias):
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

def remove_bias(array_, axis='column'):
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
