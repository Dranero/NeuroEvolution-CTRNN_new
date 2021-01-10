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
        return np.array([item for sublist in self.hidden for item in sublist])

    def get_brain_edge_weights(self):
        result = self.weight_hh[0][0]
        for i in range(1, len(self.hidden)):
            result = self.append_matrix_horizontally(result, self.append_matrix_vertically(
                np.zeros((len(self.weight_ih[i][0]), len(result[0]) - len(self.weight_ih[i][0][0]))).tolist(),
                self.weight_ih[i][0]))

            result = self.append_matrix_vertically(result, self.append_matrix_horizontally(
                np.zeros((len(result) - len(self.weight_hh[i][0]), len(self.weight_hh[i][0][0]))).tolist(),
                self.weight_hh[i][0].tolist()))
        return np.array(result)

    @staticmethod
    def append_matrix_vertically(matrix1, matrix2):
        if np.array(matrix1).size == 0:
            return matrix2
        if np.array(matrix2).size == 0:
            return matrix1
        assert (len(matrix1) == len(matrix2))
        return np.concatenate((matrix1, matrix2), 1)

    @staticmethod
    def append_matrix_horizontally(matrix1, matrix2):
        if np.array(matrix1).size == 0:
            return matrix2
        if np.array(matrix2).size == 0:
            return matrix1
        # TODO mehr überprüfen ob alles gültige Matrizen sind
        assert (len(matrix1[0]) == len(matrix2[0]))
        return np.concatenate((matrix1, matrix2), 0)

    def get_input_matrix(self):
        # TODO visualize bias
        return self.append_matrix_horizontally(
            self.weight_ih[0][0],
            np.zeros((len(self.get_brain_nodes()) - len(self.weight_ih[0][0]), len(self.weight_ih[0][0][0])))
        )

    def get_output_matrix(self):
        return self.append_matrix_vertically(
            np.zeros((len(self.weight_ho), len(self.get_brain_nodes()) - len(self.weight_ho[0]))),
            self.weight_ho
        )
