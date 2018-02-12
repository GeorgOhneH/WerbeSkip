from optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def calculate_change(self, nabla_w, nabla_b):
        change_w = self.learning_rate * nabla_w
        change_b = self.learning_rate * nabla_b
        return change_w, change_b
