# File to test if every thing works without a Exceptions
import shutil
import os

from deepnet.datasets import *
from deepnet.layers import *
from deepnet.optimizers import *
from deepnet import Network


train_data, train_labels, test_data, test_labels = load_mnist_cnn()

net = Network()

net.use_gpu = True

net.input((1, 28, 28))

net.add(ConvolutionLayer(n_filter=1, width_filter=3, height_filter=3, stride=1, zero_padding=0))
net.add(BatchNorm())
net.add(TanH())
net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
net.add(Dropout(0.75))
net.add(Flatten())
net.add(FullyConnectedLayer(3))
net.add(BatchNorm())
net.add(Sigmoid())
net.add(FullyConnectedLayer(10))
net.add(SoftMax())

optimizer = Adam(learning_rate=0.03)
net.regression(optimizer=optimizer, cost="cross_entropy")

net.fit(train_data, train_labels, validation_set=(test_data, test_labels),
        epochs=2, mini_batch_size=512, snapshot_step=2)
net.evaluate(test_data, test_labels)
path = "test"
net.save_wrong_predictions(inputs=test_data[:10], labels=test_labels[:10], directory=path, shape=(28, 28))
shutil.rmtree(path)
file = "test.h5"
net.save(file)
net.use_gpu = True
net.load(file)
net.evaluate(test_data, test_labels)
os.remove(file)
