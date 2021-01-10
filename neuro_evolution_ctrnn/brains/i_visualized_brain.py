import abc


# TODO i Think this should be abc.ABC
class IVisualizedBrain:

    @abc.abstractmethod
    def get_brain_nodes(self):
        pass

    @abc.abstractmethod
    def get_brain_edge_weights(self):
        pass

    @abc.abstractmethod
    def get_input_matrix(self):
        pass

    @abc.abstractmethod
    def get_output_matrix(self):
        pass
