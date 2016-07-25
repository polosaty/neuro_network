# -*- coding: utf-8 -*-
from neuron_factory import PerceptronNeuronFactory
from train_algorithm import Backpropagation
from net_func import Linear, BipolarSigmoid


class NeuralNetwork(object):
    '''
    *       A Neural Network constructor.
    *      - Description:      A template constructor. T is a data type, all the nodes will operate with. Create a neural network by providing it with:
    *                          @param inInputs - an integer argument - number of input neurons of newly created neural network;
    *                          @param inOutputs- an integer argument - number of output neurons of newly created neural network;
    *                          @param inNumOfHiddenLayers - an integer argument - number of hidden layers of newly created neural network, default is 0;
    *                          @param inNumOfNeuronsInHiddenLayers - an integer argument - number of neurons in hidden layers of newly created neural network ( note that every hidden layer has the same amount of neurons), default is 0;
    *                          @param inTypeOfNeuralNetwork - a const char * argument - a type of neural network, we are going to create. The values may be:
    *                          <UL>
    *                              <LI>MultiLayerPerceptron;</LI>
    *                              <LI>Default is MultiLayerPerceptron.</LI>
    *                          </UL>
    *      - Purpose:          Creates a neural network for solving some interesting problems.
    *      - Prerequisites:    The template parameter has to be picked based on your input data.
    '''
    def __init__(self,
                 inputs,
                 outputs,
                 num_of_hidden_layers=0,
                 num_of_neurons_in_hidden_layers=0,
                 type_of_neural_network="MultiLayerPerceptron"):
        self._neuron_factory = None                        # !< Member, which is responsible for creating neurons @see SetNeuronFactory
        self._training_algorithm = None                    # !< Member, which is responsible for the way the network will trained @see SetAlgorithm
        self._layers = []                                  # !< Inner representation of neural networks
        self._bias_layer = []                              # !< Container for biases

        self._inputs = inputs                              # !< Number of inputs, outputs and hidden units
        self._outputs = outputs
        self._hidden = num_of_hidden_layers

        self._mean_squared_error = 0                       # !< Mean Squared Error which is changing every iteration of the training
        self._min_mse = 0.01                               # !< The biggest Mean Squared Error required for training to stop

        # Network function's declarations for input and output neurons.

        output_neurons_func = None
        input_neurons_func = None

        # At least two layers require - input and output;
        output_layer = []
        input_layer = []

        # This block of strcmps decides what training algorithm and neuron factory we should use as well as what
        # network function every node will have.
        if type_of_neural_network == "MultiLayerPerceptron":
            self._neuron_factory = PerceptronNeuronFactory
            self._training_algorithm = Backpropagation(self)

            output_neurons_func = BipolarSigmoid
            input_neurons_func = Linear

        # Output layers creation
        for i in range(outputs):
            output_layer.append(self._neuron_factory.createOutputNeuron(output_neurons_func))

        self._layers.append(output_layer)

        # Hidden layers creation
        for i in range(num_of_hidden_layers):
            hidden_layer = []
            for j in range(num_of_neurons_in_hidden_layers):
                hidden = self._neuron_factory.createHiddenNeuron(self._layers[0], output_neurons_func)
                hidden_layer.append(hidden)

            self._bias_layer.insert(0, self._neuron_factory.createInputNeuron(self._layers[0], input_neurons_func))
            self._layers.insert(0, hidden_layer)

        # Input layers creation
        for i in range(inputs):
            input_layer.append(self._neuron_factory.createInputNeuron(self._layers[0], input_neurons_func))

        self._bias_layer.insert(0, self._neuron_factory.createInputNeuron(self._layers[0], input_neurons_func));
        self._layers.insert(0, input_layer)

        self._training_algorithm.weightsInitialization()


    '''
    *      Public method Train.
    *      - Description:      Method for training the network.
    *      - Purpose:          Trains a network, so the weights on the links adjusted in the way to be able to solve problem.
    *      - Prerequisites:
    *          @param inData   - a vector of vectors with data to train with;
    *          @param inTarget - a vector of vectors with target data;
    *                          - the number of data samples and target samples has to be equal;
    *                          - the data and targets has to be in the appropriate order u want the network to learn.
    '''
    def train(self, data, target):
        raise NotImplemented

    '''
    *      Public method GetNetResponse.
    *      - Description:      Method for actually get response from net by feeding it with data.
    *      - Purpose:          By calling this method u make the network evaluate the response for u.
    *      - Prerequisites:
    *          @param inData   - a vector data to feed with.
    '''
    def getNetResponse(self, data):
        raise NotImplemented

    '''
    *      Public method SetAlgorithm.
    *      - Description:      Setter for algorithm of training the net.
    *      - Purpose:          Can be used for dynamic change of training algorithm.
    *      - Prerequisites:
    *          @param inTrainingAlgorithm  - an existence of already created object  of type TrainAlgorithm.
    '''
    def setAlgorithm(self, training_algorithm):
        self._training_algorithm = training_algorithm

    '''
    *      Public method SetNeuronFactory.
    *      - Description:      Setter for the factory, which is making neurons for the net.
    *      - Purpose:          Can be used for dynamic change of neuron factory.
    *      - Prerequisites:
    *          @param inNeuronFactory  - an existence of already created object  of type NeuronFactory.
    '''
    def setNeuronFactory(self, neuron_factory):
        self._neuron_factory = neuron_factory

    '''
    *      Public method ShowNetworkState.
    *      - Description:      Prints current state to the standard output: weight of every link.
    *      - Purpose:          Can be used for monitoring the weights change during training of the net.
    *      - Prerequisites:    None.
    '''
    def showNetworkState(self):
        raise NotImplemented

    '''
    *      Public method GetMinMSE.
    *      - Description:      Returns the biggest MSE required to achieve during the training phase.
    *      - Purpose:          Can be used for getting the biggest MSE required to achieve during the training phase.
    *      - Prerequisites:    None.
    '''
    def getMinMSE(self):
        return self._minmse

    '''
    *      Public method SetMinMSE.
    *      - Description:      Setter for the biggest MSE required to achieve during the training phase.
    *      - Purpose:          Can be used for setting the biggest MSE required to achieve during the training phase.
    *      - Prerequisites:
    *          @param inMinMse     - double value, the biggest MSE required to achieve during the training phase.
    '''
    def setMinMSE(self, min_mse):
        self._minmse = min_mse

    '''
    *      Protected method GetLayer.
    *      - Description:      Getter for the layer by index of that layer.
    *      - Purpose:          Can be used by inner implementation for getting access to neural network's layers.
    *      - Prerequisites:
    *          @param inInd    -  an integer index of layer.
    '''
    def _getLayer(self, idx):
        return self._layers[idx]

    '''
    *      Protected method size.
    *      - Description:      Returns the number of layers in the network.
    *      - Purpose:          Can be used by inner implementation for getting number of layers in the network.
    *      - Prerequisites:    None.
    '''
    def _size(self):
        return len(self._layers)

    '''
    *      Protected method GetNumOfOutputs.
    *      - Description:      Returns the number of units in the output layer.
    *      - Purpose:          Can be used by inner implementation for getting number of units in the output layer.
    *      - Prerequisites:    None.
    '''
    def _getOutputLayer(self):
        return self._layers[-1]

    '''
    *      Protected method GetInputLayer.
    *      - Description:      Returns the input layer.
    *      - Purpose:          Can be used by inner implementation for getting the input layer.
    *      - Prerequisites:    None.
    '''
    def _getInputLayer(self):
        return self._layers[0]

    '''
    *      Protected method GetBiasLayer.
    *      - Description:      Returns the vector of Biases.
    *      - Purpose:          Can be used by inner implementation for getting vector of Biases.
    *      - Prerequisites:    None.
    '''
    def _getBiasLayer(self):
        return self._bias_layer

    '''
    *      Protected method UpdateWeights.
    *      - Description:      Updates the weights of every link between the neurons.
    *      - Purpose:          Can be used by inner implementation for updating the weights of links between the neurons.
    *      - Prerequisites:    None, but only makes sense, when its called during the training phase.
    '''
    def _updateWeights(self):
        raise NotImplemented

    '''
    *      Protected method ResetCharges.
    *      - Description:      Resets the neuron's data received during iteration of net training.
    *      - Purpose:          Can be used by inner implementation for reset the neuron's data between iterations.
    *      - Prerequisites:    None, but only makes sense, when its called during the training phase.
    '''
    def _resetCharges(self):
        raise NotImplemented

    '''
    *      Protected method AddMSE.
    *      - Description:      Changes MSE during the training phase.
    *      - Purpose:          Can be used by inner implementation for changing MSE during the training phase.
    *      - Prerequisites:
    *          @param inInd    -  a double amount of MSE to be add.
    '''
    def _addMSE(self, portion):
        self._mean_squared_error += portion

    '''
    *      Protected method GetMSE.
    *      - Description:      Getter for MSE value.
    *      - Purpose:          Can be used by inner implementation for getting access to the MSE value.
    *      - Prerequisites:    None.
    '''
    def _getMSE(self):
        return self._mean_squared_error

    '''
    *      Protected method ResetMSE.
    *      - Description:      Resets MSE value.
    *      - Purpose:          Can be used by inner implementation for resetting MSE value.
    *      - Prerequisites:    None.
    '''
    def _resetMSE(self):
        self._mean_squared_error = 0
