from deepnet.datasets import load_mnist_fc
from deepnet.layers import FullyConnectedLayer, ReLU, SoftMax, BatchNorm, Dropout
from deepnet.optimizers import Adam
from deepnet.functions.costs import QuadraticCost, CrossEntropyCost
from deepnet import Network


train_data, train_labels, test_data, test_labels = load_mnist_fc()

net = Network()

net.use_gpu = False

net.input(28 * 28)

net.add(FullyConnectedLayer(256))
net.add(BatchNorm())
net.add(ReLU())
net.add(Dropout(0.5))

net.add(FullyConnectedLayer(10))
net.add(SoftMax())

optimizer = Adam(learning_rate=0.1)
net.regression(optimizer=optimizer, cost=CrossEntropyCost())

net.fit(train_data, train_labels, validation_set=(test_data, test_labels), epochs=12, mini_batch_size=256)
net.evaluate(test_data, test_labels)
