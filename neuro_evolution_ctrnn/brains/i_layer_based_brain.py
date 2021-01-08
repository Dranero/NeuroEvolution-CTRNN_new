import abc
from brains.i_brain import IBrain, ConfigClass
from gym import Space
import numpy as np
from tools.configurations import ILayerBasedBrainCfg
from typing import List, TypeVar, Generic

LayerdConfigClass = TypeVar('LayerdConfigClass', bound=ILayerBasedBrainCfg)


class ILayerBasedBrain(IBrain, Generic[LayerdConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LayerdConfigClass):
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
        number_gates = self.get_number_gates()

        for layer in range(len(hidden_struc)):
            # Matrices for weighted input to gates all layers
            if layer == 0:
                self.weight_ih.append([[[i for i in individual[
                                                    k * input_size * hidden_struc[0]
                                                    + j * input_size:
                                                    k * input_size * hidden_struc[0]
                                                    + (j + 1) * input_size
                                                    ]] for j in range(hidden_struc[0])] for k in range(number_gates)])
                ind_index += number_gates * input_size * hidden_struc[0]
            else:
                self.weight_ih.append([[[i for i in individual[
                                                    k * hidden_struc[layer - 1] * hidden_struc[layer]
                                                    + j * hidden_struc[layer - 1]:
                                                    k * hidden_struc[layer - 1] * hidden_struc[layer]
                                                    + (j + 1) * hidden_struc[layer - 1]
                                                    ]] for j in range(hidden_struc[layer])] for k in
                                       range(number_gates)])
                ind_index += number_gates * hidden_struc[layer - 1] * hidden_struc[layer]

            # Matrices for weighted hidden to gates
            if config.each_state_one_hidden:
                self.weight_hh.append([np.diag(individual[
                                               k * hidden_struc[layer] * hidden_struc[layer]:
                                               k * hidden_struc[layer] * hidden_struc[layer] + hidden_struc[layer]])
                                       for k in range(number_gates)])
                ind_index += number_gates * hidden_struc[layer]
            else:
                self.weight_hh.append([[[i for i in individual[
                                                    k * hidden_struc[layer] * hidden_struc[layer]
                                                    + j * hidden_struc[layer]:
                                                    k * hidden_struc[layer] * hidden_struc[layer]
                                                    + (j + 1) * hidden_struc[layer]
                                                    ]] for j in range(hidden_struc[layer])] for k in
                                       range(number_gates)])
                ind_index += number_gates * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases out of individual
            # Biases for gates
            if config.use_bias:
                self.bias_h.append([[i for i in individual[
                                                ind_index + j * hidden_struc[layer]:
                                                ind_index + (j + 1) * hidden_struc[layer]
                                                ]] for j in range(number_gates)])
                ind_index += hidden_struc[layer] * number_gates
            else:
                self.bias_h.append(np.zeros((number_gates, hidden_struc[layer])).astype(np.float32))
        # for end

        # Matrix for weighted hidden to output
        self.weight_ho = [[[i for i in individual[
                                       ind_index + j * hidden_struc[len(hidden_struc) - 1]:
                                       ind_index + (j + 1) * hidden_struc[len(hidden_struc) - 1]
                                       ]] for j in range(output_size)]]
        ind_index += hidden_struc[len(hidden_struc) - 1] * output_size

    @staticmethod
    @abc.abstractmethod
    def get_number_gates():
        pass

    @classmethod
    def get_individual_size(cls, config: ILayerBasedBrainCfg, input_space: Space, output_space: Space):

        number_gates = cls.get_number_gates()
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)
        hidden_struc = config.hidden_structure

        individual_size = 0

        for layer in range(len(hidden_struc)):
            # Matrices for weighted input to gates all layers
            if layer == 0:
                individual_size += number_gates * input_size * hidden_struc[0]
            else:
                individual_size += number_gates * hidden_struc[layer] * hidden_struc[layer - 1]
            if config.each_state_one_hidden:
                individual_size += number_gates * hidden_struc[layer]
            else:
                individual_size += number_gates * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases out of individual
            # Biases for gates
            if config.use_bias:
                individual_size += hidden_struc[layer] * number_gates
        # for end

        # Matrix for weighted hidden to output
        individual_size += hidden_struc[len(hidden_struc) - 1] * output_size
        return individual_size

    def brainstep(self, ob: np.ndarray):

        layer_input = ob.astype(np.float32)

        # The input for the i-th layer is the (i-1)-th hidden feature or if i==0 the input
        # Calculated as in the PyTorch description of the LSTM:
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

        for layer in range(len(self.weight_ih)):
            if layer == 0:
                x = layer_input
            else:
                x = self.hidden[layer - 1]

            # Reset Gate
            self.hidden[layer] = self.layer_step(x, self.weight_ih[layer], self.weight_hh[layer], self.bias_h[layer],
                                                self.hidden[layer])

        return np.dot(self.weight_ho, self.hidden[len(self.hidden) - 1])

    @staticmethod
    @abc.abstractmethod
    def layer_step(layer_input: np.ndarray, wheight_ih, wheight_hh, bias_h, hidden):
        pass
