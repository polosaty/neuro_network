# -*- coding: utf-8 -*-


class NeuralLink(object):

    def __init__(self, neuron_linked_to=None, weight_to_neuron=0.0):
        self._neuron_linked_to = neuron_linked_to
        self._weight_to_neuron = weight_to_neuron
        self._weight_correction_term = 0.0
        self._error_information_term = 0.0
        self._last_translated_signal = 0.0

        def setWeight(self, in_weight):
            self._weight_to_neuron = in_weight

        def getWeight(self):
            return self._weight_to_neuron

        def setNeuronLinkedTo(self, in_neuron_linked_to):
            self._neuron_linked_to = in_neuron_linked_to

        def getNeuronLinkedTo(self):
            return self._neuron_linked_to

        def setWeightCorrectionTerm(self, in_weight_correction_term):
            self._weight_correction_term = in_weight_correction_term

        def getWeightCorrectionTerm(self):
            return self._weight_correction_term

        def updateWeight(self):
            self._weight_to_neuron = self._weight_to_neuron + self._weight_correction_term

        def getErrorInFormationTerm(self):
            return self._error_information_term

        def setErrorInFormationTerm(self, in_ei_term):
            self._error_information_term = in_ei_term

        def setLastTranslatedSignal(self, in_last_translated_signal):
            self._last_translated_signal = in_last_translated_signal

        def getLastTranslatedSignal(self):
            return self._last_translated_signal
