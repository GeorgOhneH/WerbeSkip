from Machine_Learning.Network import Network
from image_processing.image_loader import load_imgs


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_imgs()
net = Network()
net.addInputLayer(50 * 50)
net.addFullyConnectedLayer(200, activation="relu", dropout=0.8)
net.addFullyConnectedLayer(2, activation="sigmoid")
net.regression(learning_rate=0.001, cost="quadratic")
net.fit(train_data, train_labels, validation_data, validation_labels, epochs=200, mini_batch_size=20, plot=True)
net.evaluate(test_data, test_labels)
