from deepnet import Network
from image_processing.image_loader import load_imgs
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax
from deepnet.optimizers import Adam, SGD


train_data, train_labels, v_x, v_y, t_x, t_y = load_imgs(0.8)
net = Network()
net.input(52 * 52)

net.add(FullyConnectedLayer(200))
net.add(BatchNorm())
net.add(ReLU())
net.add(Dropout(0.8))

net.add(FullyConnectedLayer(200))
net.add(BatchNorm())
net.add(ReLU())
net.add(Dropout(0.8))

net.add(FullyConnectedLayer(2))
net.add(SoftMax())

optimizer = Adam(learning_rate=0.01)
net.regression(optimizer=optimizer, cost="cross_entropy")
net.fit(train_data, train_labels, validation_set=(v_x, v_y), epochs=100, mini_batch_size=128, plot=True)
net.evaluate(t_x, t_y)
net.save_wrong_predictions(t_x, t_y, "test", shape=(52, 52))
