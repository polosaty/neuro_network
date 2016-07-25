# -*- coding: utf-8 -*-
from neuron import Neuron, OutputLayerNeuronDecorator, HiddenLayerNeuronDecorator


class NeuronFactory(object):
    @staticmethod
    def createInputNeuron(in_neurons_link_to, in_net_func):
        raise NotImplemented

    @staticmethod
    def createOutputNeuron(in_net_func):
        raise NotImplemented

    @staticmethod
    def createHiddenNeuron(in_neurons_link_to, in_net_func):
        raise NotImplemented


class PerceptronNeuronFactory(NeuronFactory):
    @staticmethod
    def createInputNeuron(in_neurons_link_to, in_net_func):
        return Neuron(in_neurons_link_to, in_net_func)

    @staticmethod
    def createOutputNeuron(in_net_func):
        return OutputLayerNeuronDecorator(Neuron(in_net_func))

    @staticmethod
    def createHiddenNeuron(in_neurons_link_to, in_net_func):
        return HiddenLayerNeuronDecorator(Neuron(in_neurons_link_to, in_net_func))
