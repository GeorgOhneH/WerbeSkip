from deepnet.datasets import load_mnist_fc
from deepnet.layers import FullyConnectedLayer, ReLU, SoftMax
from deepnet.optimizers import Adam
from deepnet import Network


train_data, train_labels, test_data, test_labels = load_mnist_fc()

net = Network()

net.use_gpu = False

net.input(28 * 28)

net.add(FullyConnectedLayer(256))
net.add(ReLU())

net.add(FullyConnectedLayer(10))
net.add(SoftMax())

optimizer = Adam(learning_rate=0.1)
net.regression(optimizer=optimizer, cost="cross_entropy")

net.fit(train_data, train_labels, validation_set=(test_data, test_labels), epochs=12,
        mini_batch_size=256, plot=True, snapshot_step=10)
net.evaluate(test_data, test_labels)
net.save_wrong_predictions(inputs=test_data, labels=test_labels, directory="directory_name", shape=(28, 28))
