import numpy as np
from gym import Space
from brains.i_brain import IBrain, ConfigClass
from tools.configurations import GruCfg
from typing import List


class GruNetwork(IBrain[GruCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

        input_size = self._size_from_space(input_space)
        output_size = self._size_from_space(output_space)
        hidden_struc: List[int] = config.hidden_structure

        # initialize hidden
        # TODO is initialization with 0 good?
        self.hidden = [[0 for _ in range(i)] for i in hidden_struc]

        # initialize weights out of individual
        ind_index = 0  # laufindex
        self.weight_ih = []
        self.weight_hh = []
        self.bias_h = []

        for layer in range(len(hidden_struc)):
            # Matrices for weighted input to gates all layers
            if layer == 0:
                self.weight_ih.append([[[i for i in individual[
                                                    k * input_size * hidden_struc[0]
                                                    + j * input_size:
                                                    k * input_size * hidden_struc[0]
                                                    + (j + 1) * input_size
                                                    ]] for j in range(hidden_struc[0])] for k in range(3)])
                ind_index += 3 * input_size * hidden_struc[0]
            else:
                self.weight_ih.append([[[i for i in individual[
                                                    k * hidden_struc[layer - 1] * hidden_struc[layer]
                                                    + j * hidden_struc[layer - 1]:
                                                    k * hidden_struc[layer - 1] * hidden_struc[layer]
                                                    + (j + 1) * hidden_struc[layer - 1]
                                                    ]] for j in range(hidden_struc[layer])] for k in range(3)])
                ind_index += 3 * hidden_struc[layer - 1] * hidden_struc[layer]

            # Matrices for weighted hidden to gates
            if config.each_state_one_hidden:
                self.weight_hh.append([np.diag(individual[
                                                   k * hidden_struc[layer] * hidden_struc[layer]:
                                                   k * hidden_struc[layer] * hidden_struc[layer] + hidden_struc[layer]])
                                       for k in range(3)])
                ind_index += 3 * hidden_struc[layer]
            else:
                self.weight_hh.append([[[i for i in individual[
                                                    k * hidden_struc[layer] * hidden_struc[layer]
                                                    + j * hidden_struc[layer]:
                                                    k * hidden_struc[layer] * hidden_struc[layer]
                                                    + (j + 1) * hidden_struc[layer]
                                                    ]] for j in range(hidden_struc[layer])] for k in range(3)])
                ind_index += 3 * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases out of individual
            # Biases for gates
            if config.use_bias:
                self.bias_h.append([[i for i in individual[
                                                ind_index + j * hidden_struc[layer]:
                                                ind_index + (j + 1) * hidden_struc[layer]
                                                ]] for j in range(3)])
                ind_index += hidden_struc[layer] * 3
            else:
                self.bias_h.append(np.zeros((3, hidden_struc[layer])).astype(np.float32))
        # for end

        # Matrix for weighted hidden to output
        self.weight_ho = [[[i for i in individual[
                                       ind_index + j * hidden_struc[len(hidden_struc) - 1]:
                                       ind_index + (j + 1) * hidden_struc[len(hidden_struc) - 1]
                                       ]] for j in range(output_size)]]
        ind_index += hidden_struc[len(hidden_struc) - 1] * output_size

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def brainstep(self, ob: np.ndarray):

        input = ob.astype(np.float32)

        # The input for the i-th layer is the (i-1)-th hidden feature or if i==0 the input
        # Calculated as in the PyTorch description of the LSTM:
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

        for layer in range(len(self.weight_ih)):
            if layer == 0:
                x = input
            else:
                x = self.hidden[layer - 1]
            # Reset Gate
            r_t = self.sigmoid(np.dot(self.weight_ih[layer][0], x)
                               + np.dot(self.weight_hh[layer][0], self.hidden[layer])
                               + self.bias_h[layer][0])

            # Update Gate
            z_t = self.sigmoid(np.dot(self.weight_ih[layer][1], x)
                               + np.dot(self.weight_hh[layer][1], self.hidden[layer])
                               + self.bias_h[layer][1])

            # New Gate
            n_t = np.tanh(np.dot(self.weight_ih[layer][2], x)
                          + np.dot(r_t, np.dot(self.weight_hh[layer][2], self.hidden[layer])
                                   + self.bias_h[layer][2]))

            self.hidden[layer] = np.multiply(1 - z_t, n_t) + np.multiply(z_t, self.hidden[layer])

        return np.dot(self.weight_ho, self.hidden[len(self.hidden) - 1])

    @classmethod
    def get_individual_size(cls, config: GruCfg, input_space: Space, output_space: Space):

        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)
        hidden_struc = config.hidden_structure

        individual_size = 0

        for layer in range(len(hidden_struc)):
            # Matrices for weighted input to gates all layers
            if layer == 0:
                individual_size += 3 * input_size * hidden_struc[0]
            else:
                individual_size += 3 * hidden_struc[layer] * hidden_struc[layer - 1]
            if config.each_state_one_hidden:
                individual_size += 3 * hidden_struc[layer]
            else:
                individual_size += 3 * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases out of individual
            # Biases for gates
            if config.use_bias:
                individual_size += hidden_struc[layer] * 3
        # for end

        # Matrix for weighted hidden to output
        individual_size += hidden_struc[len(hidden_struc) - 1] * output_size
        return individual_size
