import numpy as np
from gym import Space

from neuro_evolution_ctrnn.brains.i_brain import IBrain, ConfigClass
from neuro_evolution_ctrnn.tools.configurations import GruCfg


class GruNetwork(IBrain[GruCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

        self.config = config
        self.input_space = input_space

        self.input_size = self._size_from_space(input_space)
        self.output_size = self._size_from_space(output_space)
        self.hidden = [0 for _ in range(self.output_size)]

        # initialize weights out of individual
        self.weight_ih_l0 = [[[i for i in individual[
                                          k * self.input_size * self.input_size + j * self.input_size:
                                          k * self.input_size * self.input_size + (j + 1) * self.input_size
                                          ]]
                              for j in range(self.input_size)] for k in range(3)]
        self.weight_hh_l0 = [[[i for i in individual[
                                          (3 + k) * self.input_size * self.input_size + j * self.input_size:
                                          (3 + k) * self.input_size * self.input_size + (j + 1) * self.input_size
                                          ]]
                              for j in range(self.input_size)] for k in range(3)]
        # initialize biases out of individual
        if config.use_bias:
            self.bias_ih_l0 = [[i for i in individual[
                                           6 * self.input_size * self.input_size:
                                           6 * self.input_size * self.input_size + self.output_size
                                           ]]
                               for _ in range(3)]
            self.bias_hh_l0 = [[i for i in individual[
                                           6 * self.input_size * self.input_size + self.output_size:
                                           6 * self.input_size * self.input_size + 2 * self.output_size
                                           ]]
                               for _ in range(3)]
        else:
            self.bias_ih_l0 = self.bias_hh_l0 = np.zeros((3, self.output_size)).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def step(self, ob: np.ndarray):

        if self.config.normalize_input:
            ob = self._normalize_input(ob, self.input_space, self.config.normalize_input_target)

        x = ob.astype(np.float32)

        # The input for the i-th layer is the (i-1)-th hidden feature or if i==0 the input
        # Calculated as in the PyTorch description of the LSTM:
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

        # Reset Gate
        r_t = self.sigmoid(np.dot(self.weight_ih_l0[0], x)
                           + self.bias_ih_l0[0]
                           + np.dot(self.weight_hh_l0[0], self.hidden)
                           + self.bias_hh_l0[0])

        # Update Gate
        z_t = self.sigmoid(np.dot(self.weight_ih_l0[1], x)
                           + self.bias_ih_l0[1]
                           + np.dot(self.weight_hh_l0[1], self.hidden)
                           + self.bias_hh_l0[1])

        # New Gate
        n_t = np.tanh(np.dot(self.weight_ih_l0[2], x)
                      + self.bias_ih_l0[2]
                      + np.dot(r_t, np.dot(self.weight_hh_l0[2], self.hidden)
                               + self.bias_hh_l0[2]))

        self.hidden = np.multiply(1 - z_t, n_t) + np.multiply(z_t, self.hidden)

        return self.hidden

    @classmethod
    def get_individual_size(cls, config: GruCfg, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        individual_size = 0

        # Calculate the number of weights as depicted in https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        individual_size += 3 * output_size * (input_size + output_size)

        if config.use_bias:
            individual_size += 6 * output_size

        return individual_size
