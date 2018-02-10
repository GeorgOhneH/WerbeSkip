class Layer(object):
    def init(self, neurons_before):
        return neurons_before

    def forward(self, a):
        return a

    def forward_backpropagation(self, a):
        return a

    def make_delta(self, delta):
        return delta

    def adjust_weights(self, factor):
        pass
