# -*- coding: utf-8 -*-
from math import exp  # expm1


class NetworkFunction(object):

    @classmethod
    def process(self, in_param):
        raise NotImplemented

    @classmethod
    def derivative(self, in_param):
        raise NotImplemented


class Linear(NetworkFunction):

    @classmethod
    def process(self, in_param):
        return in_param

    @classmethod
    def derivative(self, in_param):
        return 0


class Sigmoid(NetworkFunction):

    @classmethod
    def process(self, in_param):
        return (1 / (1 + exp(-in_param)))

    @classmethod
    def derivative(self, in_param):
        return self.process(in_param) * (1 - self.process(in_param))


class BipolarSigmoid (NetworkFunction):

    @classmethod
    def process(self, in_param):
        return 2 / (1 + exp(- in_param)) - 1

    @classmethod
    def derivative(self, in_param):
        return 0.5 * (1 + self.process(in_param)) * (1 - self.process(in_param))

