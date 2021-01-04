import numpy as np
from gym import Space

from neuro_evolution_ctrnn.brains.i_brain import IBrain, ConfigClass
from neuro_evolution_ctrnn.tools.configurations import ElmanCfg


class ElmanNetwork(IBrain[ElmanCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)
        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        self.config = config
        self.input_space: Space = input_space
        self.output_space: Space = output_space
        # gets overwritten by hidden Layer. First step no data -> initialized with 0
        self.state_layer = [0] * config.hidden_space

        in_size = self._size_from_space(input_space)
        out_size = self._size_from_space(output_space)
        hi_size = self.config.hidden_space
        # row after row
        # TODO this should somehow be implemented cleaner
        self.M_input_hidden = [[i for i in individual[j * in_size: j * in_size + in_size]] for j in range(hi_size)]
        self.M_hidden_output = [[i for i in individual[
                                            in_size * hi_size + j * hi_size:
                                            in_size * hi_size + j * hi_size + hi_size
                                            ]] for j in range(out_size)]
        if self.config.each_state_one_hidden:
            self.M_state_hidden = [[i] for i in individual[
                                                in_size * hi_size + hi_size * out_size:
                                                in_size * hi_size + hi_size * out_size + hi_size:
                                                ]]
        else:
            self.M_state_hidden = [[i for i in individual[
                                               in_size * hi_size + hi_size * out_size + j * hi_size:
                                               in_size * hi_size + hi_size * out_size + j * hi_size + hi_size
                                               ]] for j in range(config.hidden_space)]

    def step(self, ob: np.ndarray):
        # Annahme: ob sind Inputwerte
        assert len(ob) == self._size_from_space(self.input_space)
        hidden = np.dot(self.M_input_hidden, ob) + np.dot(self.M_state_hidden, self.state_layer)
        self.state_layer = hidden
        return np.dot(self.M_hidden_output, hidden)

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        # Only the three matrices will be optimized
        indvidual_size = 0
        # M_input_hidden
        indvidual_size += cls._size_from_space(input_space) * config.hidden_space
        # M_hidden_output
        indvidual_size += config.hidden_space * cls._size_from_space(output_space)
        # M_state_hidden
        if config.each_state_one_hidden:
            indvidual_size += config.hidden_space
        else:
            indvidual_size += config.hidden_space * config.hidden_space
        return indvidual_size
