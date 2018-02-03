from Machine_Learning.Network import Network
from image_processing.image_loader import load_imgs


train_data, train_labels, test_data, test_labels = load_imgs()
net = Network()
net.addInputLayer(50 * 50)
net.addFullyConnectedLayer(50, activation="relu")
net.addFullyConnectedLayer(50, activation="relu")
net.addFullyConnectedLayer(2, activation="sigmoid")
net.regression(learning_rate=0.01, cost="quadratic")
net.fit(train_data, train_labels, epochs=50, mini_batch_size=10, plot=True)
net.accuracy(test_data, test_labels)
