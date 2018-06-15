class Generator(object):
    def __init__(self, epochs, mini_batch_size):
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs

    def __next__(self):
        raise NotImplemented

    def __len__(self):
        return 1
