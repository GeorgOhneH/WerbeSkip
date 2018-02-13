from Machine_Learning.Network import Network
from image_processing.image_loader import load_imgs
from PIL import Image
import numpy as np


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_imgs()
net = Network()
net.addInputLayer(52 * 52)
net.addFullyConnectedLayer(100, activation="relu")
net.addDropout(0.8)
net.addFullyConnectedLayer(2, activation="sigmoid")
net.regression(learning_rate=0.01, cost="quadratic")
net.fit(train_data, train_labels, validation_data, validation_labels, epochs=40, mini_batch_size=20, plot=True, snapshot_step=100)
img_dataer = net.evaluate(test_data, test_labels)

# print(len(img_dataer))
# for index, data in enumerate(img_dataer):
#     img_data, label = data[0], data[1]
#     img_data = np.array([int(x*255) for x in img_data])
#     img_data = img_data.reshape((50, 50)).astype('uint8')
#     img = Image.fromarray(img_data)
#     img.save("C:\Jetbrains\PyCharm\WerbeSkip\Zattoo\prosieben\\test\\{}{}".format(index, label))
