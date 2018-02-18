class Optimizer(object):
    """
    This is the base layer for the optimizer
    It is non-functional
    """
    def __init__(self, learning_rate):
        """
        :param learning_rate: unsigned int
        """
        self.learning_rate = learning_rate

    def __str__(self):
        return self.__class__.__name__

    def calculate_change(self, *nablas):
        """
        Calculates the change by how much the parameter need to change
        :param nablas: list of parameters: ndarray
        :return: list of parameter: same length and shape as the input: ndarray
        """
        pass
