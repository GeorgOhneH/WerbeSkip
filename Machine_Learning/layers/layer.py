class Layer(object):
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation
        self.biases = None
        self.weights = None
        self.nabla_b = None
        self.nabla_w = None
        self.z = None
        self.before_a = None
        self.a = None

    def init(self, neurons_before):
        pass

    # Input Matrix Output Matrix
    def forward(self, a):
        pass

    def forward_backpropagation(self, a):
        pass

    def make_delta(self, delta, last_weights):
        pass

    def adjust_weights(self, factor):
        pass
