from .optimizer import Optimizer


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = 0
        self.velocity_b = 0

    def calculate_change(self, nabla_w, nabla_b):
        self.velocity_w = self.momentum * self.velocity_w + self.learning_rate * nabla_w
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * nabla_b
        return self.velocity_w, self.velocity_b
