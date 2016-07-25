# -*- coding: utf-8 -*-
from net_func import Linear


class Neuron(object):
    def __init__(self, in_net_func=None, in_links_to_neurons=None):
        self._net_func = in_net_func or Linear()
        self._input_links = []
        self._links_to_neurons = in_links_to_neurons or []
        self._sum_of_charges = 0.0

    def at(self, i):
        return self._links_to_neurons[i]

    def getLinksToNeurons(self):
        return self._links_to_neurons

    def setLinkToNeuron(self, in_neural_link):
        self._links_to_neurons.push(in_neural_link)

    def input(self, in_input_data):
        self._sum_of_charges += in_input_data

    def fire(self):
        raise NotImplemented

    def GetNumOfLinks(self):
        return len(self._links_to_neurons)

    def getSumOfCharges(self):
        raise NotImplemented

    def resetSumOfCharges(self):
        self._sum_of_charges = 0.0

    def process(self):
        return self._net_func.process(self._sum_of_charges)

    def process_in(self, in_arg):
        return self._net_func.process(in_arg)

    def derivative(self):
        return self._net_func.derivative(self._sum_of_charges)

    def setInputLink(self, in_link):
        self._input_links.append(in_link)

    def getInputLink(self):
        return self._input_links

    def performTrainingProcess(self, in_target):
        raise NotImplemented

    def performWeightsUpdating(self):
        raise NotImplemented

    def showNeuronState(self):
        raise NotImplemented


class OutputLayerNeuronDecorator(Neuron):
    def __init__(self, neuron):
        self._neuron = neuron
        self._output_charge = 0.0

    def getLinksToNeurons(self):
        return self._neuron.getLinksToNeurons()

    def at(self, in_index_of_neural_link):
        return (self._neuron.at(in_index_of_neural_link))

    def setLinkToNeuron(self, in_neural_link):
        self._neuron.setLinkToNeuron(in_neural_link)

    def getSumOfCharges(self):
        return self._neuron.getSumOfCharges()

    def resetSumOfCharges(self):
        self._neuron.resetSumOfCharges()

    def input(self, in_input_data):
        self._neuron.input(in_input_data)

    # def fire(self):
    #     raise NotImplemented

    def getNumOfLinks(self):
        return self._neuron.getNumOfLinks()

    def process(self):
        return self._neuron.process()

    def process_in(self, in_arg):
        return self._neuron.process(in_arg)

    def derivative(self):
        return self._neuron.derivative()

    def setInputLink(self, in_link):
        self._neuron.setInputLink(in_link)

    def getInputLink(self):
        return self._neuron.getInputLink()

    # def performTrainingProcess(self, in_target):
    #     raise NotImplemented

    # def performWeightsUpdating(self):
    #     raise NotImplemented

    def showNeuronState(self):
        self._neuron.showNeuronState()


class HiddenLayerNeuronDecorator(OutputLayerNeuronDecorator):
    def __init__(self, neuron):
        self._neuron = neuron


