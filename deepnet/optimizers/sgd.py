from deepnet.optimizers import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent is one of the simplest optimizer

    It just multiplies the error with the leaning rate
    """
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def calculate_change(self, *nablas):
        changes = []
        for nabla in nablas:
            changes.append(self.learning_rate * nabla)
        return changes
