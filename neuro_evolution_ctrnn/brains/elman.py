import numpy as np
from gym import Space
from brains.i_brain import ConfigClass, IBrain
from tools.configurations import ElmanCfg
from brains.i_layer_based_brain import ILayerBasedBrain
import torch
import torch.nn as nn


class ElmanNetwork(ILayerBasedBrain[ElmanCfg]):

    @staticmethod
    def get_number_gates():
        return 1

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        result = np.tanh(
            np.dot(weight_ih[0], layer_input) +
            np.dot(weight_hh[0], hidden) +
            bias_h[0])
        return [result, result]

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

    def get_brain_nodes(self):
        return np.array([item for sublist in self.hidden for item in sublist])

    def get_brain_edge_weights(self):
        result: np.array = self.weight_hh[0][0]
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


class ElmanPytorch(IBrain):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        with torch.no_grad():
            #only one layer
            self.elman = nn.RNN(
                IBrain._size_from_space(input_space), IBrain._size_from_space(output_space),
                1, bias=config.use_bias)

    def brainstep(self, ob: np.ndarray):
        with torch.no_grad():
            # Input requires the form (seq_len, batch, input_size)
            out, self.hidden = self.elman(torch.from_numpy(ob.astype(np.float32)).view(1, 1, -1), self.hidden)
            return out.view(IBrain._size_from_space(self.output_space)).numpy()

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        pass
