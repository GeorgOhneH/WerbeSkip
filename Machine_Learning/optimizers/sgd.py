from optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def calculate_change(self, *nablas):
        changes = []
        for nabla in nablas:
            changes.append(self.learning_rate * nabla)
        return changes
