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

        # initialize weights out of individual
        ind_index = 0  # laufindex
        self.weight_ih = []
        self.weight_hh = []
        self.bias_h = []
        self.hidden = []
        self.layer_output = []
        number_gates = self.get_number_gates()

        for layer in range(len(hidden_struc)):
            # Matrices for weighted input to gates all layers

            if layer == 0:
                self.weight_ih.append(
                    np.array(individual[ind_index:ind_index + number_gates * hidden_struc[layer] * input_size])
                        .reshape((number_gates, hidden_struc[layer], input_size))
                )
                ind_index += number_gates * input_size * hidden_struc[0]
            else:
                self.weight_ih.append(np.full(
                    (number_gates, hidden_struc[layer], hidden_struc[layer - 1]),
                    individual[ind_index:ind_index + number_gates * hidden_struc[layer] * hidden_struc[layer - 1]]))
                ind_index += number_gates * hidden_struc[layer - 1] * hidden_struc[layer]

            # Matrices for weighted hidden to gates
            if config.diagonal_hidden_to_hidden:
                self.weight_hh.append([np.diag(individual[
                                               ind_index + k * hidden_struc[layer] * hidden_struc[layer]:
                                               ind_index + k * hidden_struc[layer] * hidden_struc[layer] + hidden_struc[
                                                   layer]])
                                       for k in range(number_gates)])
                ind_index += number_gates * hidden_struc[layer]
            else:
                self.weight_hh.append(np.full(
                    (number_gates, hidden_struc[layer], hidden_struc[layer]),
                    individual[ind_index:ind_index + number_gates * hidden_struc[layer] * hidden_struc[layer]]))
                ind_index += number_gates * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases out of individual
            # Biases for gates
            if config.use_bias:
                self.bias_h.append(np.full(
                    (number_gates, hidden_struc[layer]),
                    individual[ind_index: ind_index + hidden_struc[layer] * number_gates]))
                ind_index += hidden_struc[layer] * number_gates
            else:
                self.bias_h.append(np.zeros((number_gates, hidden_struc[layer])).astype(np.float32))
            # initialize hidden
            if config.optimize_initial_hidden_values:
                self.hidden.append(
                    np.full((hidden_struc[layer]), individual[ind_index:ind_index + hidden_struc[layer]]))
                ind_index += hidden_struc[layer]
            else:
                self.hidden.append(np.zeros((hidden_struc[layer])))
            self.layer_output.append(np.zeros((hidden_struc[layer])))
        # for end

        # Matrix for weighted hidden to output
        self.weight_ho = np.array(
            individual[ind_index:ind_index + hidden_struc[len(hidden_struc) - 1] * output_size]).reshape(
            (output_size, hidden_struc[len(hidden_struc) - 1])
        )
        ind_index += hidden_struc[len(hidden_struc) - 1] * output_size

        assert ind_index == len(individual)

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
            if config.diagonal_hidden_to_hidden:
                individual_size += number_gates * hidden_struc[layer]
            else:
                individual_size += number_gates * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases out of individual
            # Biases for gates
            if config.use_bias:
                individual_size += hidden_struc[layer] * number_gates
            # Hidden values
            if config.optimize_initial_hidden_values:
                individual_size += hidden_struc[layer]
        # for end

        # Matrix for weighted hidden to output
        individual_size += hidden_struc[len(hidden_struc) - 1] * output_size
        return individual_size

    def brainstep(self, ob: np.ndarray):

        layer_input = ob

        for layer in range(len(self.hidden)):
            if layer == 0:
                x = layer_input
            else:
                x = self.layer_output[layer - 1]

            layer_result = self.layer_step(x, self.weight_ih[layer], self.weight_hh[layer], self.bias_h[layer],
                                           self.hidden[layer])
            self.hidden[layer] = layer_result[0]
            self.layer_output[layer] = layer_result[1]
        return np.dot(self.weight_ho, self.layer_output[len(self.layer_output) - 1])

    # Returns a list with two elements.
    # The first element is the calculated new hidden cell state, the second is the layer output
    @staticmethod
    @abc.abstractmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        pass
