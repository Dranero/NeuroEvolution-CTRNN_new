from abc import ABC

import numpy as np
from gym import Space

from brains.i_brain import IBrain, ConfigClass
from tools.configurations import ElmanCfg
from brains.i_layer_based_brain import ILayerBasedBrain


class ElmanNetwork(ILayerBasedBrain[ElmanCfg]):

    @staticmethod
    def get_number_gates():
        return 1

    @staticmethod
    def layer_step(layer_input: np.ndarray, wheight_ih, wheight_hh, bias_h, hidden):
        return np.tanh(
            np.dot(wheight_ih, layer_input) +
            np.dot(wheight_hh, hidden) +
            bias_h)

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)
