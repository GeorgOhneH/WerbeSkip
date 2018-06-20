from layers.layer import Layer

import numpywrapper as np
import numpy


class Dropout(Layer):
    """
    Dropout deactivates random neurons, so the network is forced to use
    different neurons and neurons can't become to dominant

    It's mainly used to prevent overfitting

    It takes one argument
    :param dropout: flout
        the number must be between 0 and 1
        0 :deactivates all neurons
        1: doesn't has an effect
        0.5: deactivates neurons by a 50% chance
    """
    def __init__(self, dropout):
        """
        :param dropout: flout
            must be number between 0 and 1
        """
        self.dropout = dropout
        self.dropout_mask = None
        self.z = None
        self.a = None

    def __str__(self):
        return "{}: dropout: {}".format(super(Dropout, self).__str__(), self.dropout)

    def forward(self, a):
        """
        Multiples the activation by the dropout
        This is need because so are the sum of the weights
        the same, as they were in training
        """
        return a * self.dropout

    def forward_backpropagation(self, a):
        """Deactivates neurons by multiplying them by 0"""
        self.dropout_mask = np.asarray(numpy.random.binomial(1, self.dropout, size=a.shape))
        return a * self.dropout_mask

    def make_delta(self, delta):
        """Applies the mask"""
        return delta * self.dropout_mask

