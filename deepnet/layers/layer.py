class Layer(object):
    """
    The basic layer class. It is not functional but provides a base for the
    network, so if layers don`t use all the methods the forward and backward path
     still works by giving the same values back it received
    """
    def __str__(self):
        return self.__class__.__name__

    def init(self, neurons_before, optimizer):
        """
        Initial the layer
        :param neurons_before: int
        :param optimizer: optimizer class
        :return: number of neurons of this : int
        """

        return neurons_before

    def forward(self, a):
        """
        Normal forward path of the network
        :param a: ndarray
        :return: ndarray
        """

        return a

    def forward_backpropagation(self, a):
        """
        Forward path if the network is training
        :param a: ndarray
        :return: ndarray
        """
        a = self.forward(a)
        return a

    def make_delta(self, delta):
        """
        Backward path of the network
        :param delta: ndarray
        :return: ndarray
        """

        return delta

    def adjust_parameters(self, mini_batch_size):
        """
        Adjust the parameters of the layer by using the optimizer
        :param mini_batch_size:
        :return: None
        """

        pass

    def save(self):
        return []

    def load(self, array):
        pass
