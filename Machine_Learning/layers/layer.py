class Layer(object):
    def __init__(self, neurons, activation, dropout):
        self.neurons = neurons
        self.activation = activation
        self.dropout = dropout
        self.dropout_mask = None
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

    def make_first_delta(self, cost, y):
        pass

    def make_next_delta(self, delta, last_weights):
        pass