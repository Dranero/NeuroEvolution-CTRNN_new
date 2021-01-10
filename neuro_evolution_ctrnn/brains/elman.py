import numpy as np
from gym import Space
from brains.i_brain import IBrain, ConfigClass
from tools.configurations import ElmanCfg
from brains.i_layer_based_brain import ILayerBasedBrain

from brains.i_visualized_brain import IVisualizedBrain
from typing import List


class ElmanNetwork(ILayerBasedBrain[ElmanCfg], IVisualizedBrain):

    @staticmethod
    def get_number_gates():
        return 1

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        return np.tanh(
            np.dot(weight_ih[0], layer_input) +
            np.dot(weight_hh[0], hidden) +
            bias_h[0])

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

    def get_brain_nodes(self):
        return np.ndarray.flatten(np.array(self.hidden))

    def get_brain_edge_weights(self):
        result = self.weight_hh[0][0]
        for i in range(1, len(self.hidden)):
            result_width = len(result[0])
            result = self.append_matrix_vertically(result, self.append_matrix_horizontally(
                np.zeros((len(result) - len(self.weight_ih[i][0]), len(self.weight_ih[i][0][0]))),
                self.weight_ih[i][0]))
            result = self.append_matrix_horizontally(result, self.append_matrix_vertically(
                np.zeros((len(self.weight_hh[i][0]), result_width), self.weight_hh[i][0])))
        return result

    @staticmethod
    def append_matrix_vertically(matrix1, matrix2):
        if not matrix1.any():
            return matrix2
        if not matrix2:
            return matrix1
        assert (len(matrix1) == len(matrix2))
        result: List[List[int]] = []
        for i in range(len(matrix1)):
            result.append(matrix1[i] + matrix2[i])
        return result

    @staticmethod
    def append_matrix_horizontally(matrix1, matrix2):
        if not matrix1:
            return matrix2
        if not matrix2:
            return matrix1
        # TODO mehr überprüfen ob alles gültige Matrizen sind
        assert (len(matrix1[0]) == len(matrix2[0]))
        return matrix1 + matrix2

    def get_input_matrix(self):
        return self.append_matrix_horizontally(
            self.weight_ih[0][0],
            np.zeros((len(self.get_brain_nodes()) - len(self.weight_ih[0][0]), len(self.weight_ih[0][0][0])))
        )

    def get_output_matrix(self):
        return self.append_matrix_vertically(
            np.zeros((len(self.weight_ho), len(self.get_brain_nodes()) - len(self.weight_ho[0]))),
            self.weight_ho
        )
