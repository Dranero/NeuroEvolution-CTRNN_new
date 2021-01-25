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
        # A complete matrix for the weighted input values
        # Is used to draw the lines in the visualization
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
        return np.concatenate((matrix1, matrix2), 1)

    @staticmethod
    def append_matrix_horizontally(matrix1, matrix2):
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


class ElmanPytorch(nn.Module, IBrain):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        nn.Module.__init__(self)
        IBrain.__init__(self, input_space, output_space, individual, config)
        self.output_size = IBrain._size_from_space(self.output_space)
        self.hidden = (
            torch.zeros(config.num_layers, 1, config.hidden_size)
        )

        with torch.no_grad():
            self.elman = nn.RNN(
                input_size=IBrain._size_from_space(input_space), hidden_size=config.hidden_size,
                num_layers=config.num_layers, bias=config.use_bias)
        current_index = 0
        for i in range(config.num_layers):
            attr_weight_ih_li = "weight_ih_l{}".format(i)
            attr_weight_hh_li = "weight_hh_l{}".format(i)

            weight_ih_li = getattr(self.elman, attr_weight_ih_li)
            weight_hh_li = getattr(self.elman, attr_weight_hh_li)

            weight_ih_li_size = np.prod(weight_ih_li.size())
            weight_hh_li_size = np.prod(weight_hh_li.size())

            weight_ih_li.data = torch.from_numpy(np.asarray(
                individual[current_index: current_index + weight_ih_li_size])).view(weight_ih_li.size()).float()
            current_index += weight_ih_li_size

            weight_hh_li.data = torch.from_numpy(np.asarray(
                individual[current_index: current_index + weight_hh_li_size])).view(weight_hh_li.size()).float()
            current_index += weight_hh_li_size

            if config.use_bias:
                attr_bias_ih_li = "bias_ih_l{}".format(i)
                attr_bias_hh_li = "bias_hh_l{}".format(i)

                bias_ih_li = getattr(self.elman, attr_bias_ih_li)
                bias_hh_li = getattr(self.elman, attr_bias_hh_li)

                bias_ih_li_size = bias_ih_li.size()[0]
                bias_hh_li_size = bias_hh_li.size()[0]

                bias_ih_li.data = torch.from_numpy(
                    np.asarray(individual[current_index: current_index + bias_ih_li_size])).float()
                current_index += bias_ih_li_size

                bias_hh_li.data = torch.from_numpy(
                    np.asarray(individual[current_index: current_index + bias_hh_li_size])).float()
                current_index += bias_hh_li_size

        assert current_index == len(individual)

    def brainstep(self, ob: np.ndarray):
        with torch.no_grad():
            # Input requires the form (seq_len, batch, input_size)
            out, self.hidden = self.elman(torch.from_numpy(ob.astype(np.float32)).view(1, 1, -1), self.hidden)
            return out.view(self.output_size).numpy()

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        index = 0
        # size of the learnable input-hidden weights
        index += config.hidden_size * IBrain._size_from_space(input_space) * config.num_layers
        # size of the learnable hidden-hidden weights
        index += config.hidden_size * config.hidden_size * config.num_layers

        if config.use_bias:
            index += 2 * config.hidden_size * config.num_layers
        return index

    def get_brain_nodes(self):
        raise NotImplementedError

    def get_brain_edge_weights(self):
        raise NotImplementedError

    def get_input_matrix(self):
        raise NotImplementedError

    def get_output_matrix(self):
        raise NotImplementedError
