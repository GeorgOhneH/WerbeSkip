from deepnet.datasets.mnist_loader import load_mnist
from deepnet.layers import FullyConnectedLayer, ReLU, SoftMax
from deepnet.optimizers import Adam
from deepnet import Network


train_data, train_labels, test_data, test_labels = load_mnist()

net = Network()

net.use_gpu = False

net.input(28 * 28)

net.add(FullyConnectedLayer(128))
net.add(ReLU())

net.add(FullyConnectedLayer(10))
net.add(SoftMax())

optimizer = Adam(learning_rate=0.03)
net.regression(optimizer=optimizer, cost="cross_entropy")

net.fit(train_data, train_labels, validation_set=(test_data, test_labels),
        epochs=12, mini_batch_size=256, plot=False, snapshot_step=10)
net.evaluate(test_data, test_labels)
