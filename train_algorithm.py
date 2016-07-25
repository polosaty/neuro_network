# -*- coding: utf-8 -*-


class TrainAlgorithm(object):
    def Train(in_data, in_target):
        raise NotImplemented

    def WeightsInitialization():
        raise NotImplemented


class Hebb(TrainAlgorithm):
    def __init__(self, neural_network):
        self._neural_network = neural_network


class Backpropagation(TrainAlgorithm):
    def __init__(self, neural_network):
        self._neural_network = neural_network

    def _nguyenWidrowWeightsInitialization(self):
        raise NotImplemented

    def _commonInitialization(self):
        raise NotImplemented

