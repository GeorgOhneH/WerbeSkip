class Layer(object):
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation
        self.biases = None
        self.weights = None
        self.nabla_b = None
        self.nabla_w = None
        self.z = None
        self.a = None

    def init(self, neurons_before):
        return neurons_before

    # Input Matrix Output Matrix
    def forward(self, a):
        return a

    def forward_backpropagation(self, a):
        return a

    def make_delta(self, delta):
        return delta

    def adjust_weights(self, factor):
        pass
