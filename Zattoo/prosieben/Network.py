from Machine_Learning import Network
from image_processing.image_loader import load_imgs
from Machine_Learning.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax
from Machine_Learning.optimizers import Adam, SGD


train_data, train_labels, v_x, v_y, t_x, t_y = load_imgs()
print(train_data.shape)
net = Network()
net.input(52 * 52)

net.add(FullyConnectedLayer(20))
net.add(BatchNorm())
net.add(ReLU())
net.add(Dropout(0.8))

net.add(FullyConnectedLayer(200))
net.add(BatchNorm())
net.add(ReLU())
net.add(Dropout(0.8))

net.add(FullyConnectedLayer(2))
net.add(SoftMax())

optimizer = SGD(learning_rate=0.005)
net.regression(optimizer=optimizer, cost="cross_entropy")
net.fit(train_data, train_labels, v_x, v_y, epochs=2, mini_batch_size=20, plot=True)
net.evaluate(t_x, t_y)

# print(len(img_dataer))
# for index, data in enumerate(img_dataer):
#     img_data, label = data[0], data[1]
#     img_data = np.array([int(x*255) for x in img_data])
#     img_data = img_data.reshape((50, 50)).astype('uint8')
#     img = Image.fromarray(img_data)
#     img.save("C:\Jetbrains\PyCharm\WerbeSkip\Zattoo\prosieben\\test\\{}{}".format(index, label))
