import numpy as np
from gym import Space
from brains.i_brain import IBrain, ConfigClass
from tools.configurations import GruCfg
from brains.i_layer_based_brain import ILayerBasedBrain


class GruNetwork(ILayerBasedBrain[GruCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def layer_step(layer_input, weight_ih, weight_hh, bias_h, hidden):
        r_t = GruNetwork.sigmoid(np.dot(weight_ih[0], layer_input)
                           + np.dot(weight_hh[0], hidden)
                           + bias_h[0])

        # Update Gate
        z_t = GruNetwork.sigmoid(np.dot(weight_ih[1], layer_input)
                           + np.dot(weight_hh[1], hidden)
                           + bias_h[1])

        # New Gate
        n_t = np.tanh(np.dot(weight_ih[2], layer_input)
                      + np.dot(r_t, np.dot(weight_hh[2], hidden)
                               + bias_h[2]))

        return np.multiply(1 - z_t, n_t) + np.multiply(z_t, hidden)

    @staticmethod
    def get_number_gates():
        return 3
