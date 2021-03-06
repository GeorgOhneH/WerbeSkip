class Optimizer(object):
    """
    all after https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f
    This is the base layer for the optimizer
    It is non-functional
    """
    def __init__(self, learning_rate):
        """
        :param learning_rate: unsigned int
        """
        self.learning_rate = learning_rate

    def __str__(self):
        return "{}: learning rate: {}".format(self.__class__.__name__, self.learning_rate)

    def calculate_change(self, *nablas):
        """
        Calculates the change by how much the parameter need to change
        :param nablas: list of parameters: ndarray
        :return: list of parameter: same length and shape as the input: ndarray
        """
        pass
