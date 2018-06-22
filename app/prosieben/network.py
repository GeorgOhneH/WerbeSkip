from deepnet import Network
from app.image_processing.image_loader import load_imgs
from app.image_processing.generator import TrainGenerator
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax
from deepnet.optimizers import Adam


if __name__ == "__main__":
    generator = TrainGenerator(epochs=1, mini_batch_size=128, padding=10, n_workers=1)

    _, v_x, v_y, t_x, t_y = load_imgs(0.8)
    print(v_x.shape, v_y.shape)
    net = Network()
    net.use_gpu = True
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
    net.fit_generator(generator, plot=False)
    net.evaluate(t_x, t_y)
    net.save_wrong_predictions(t_x, t_y, "wrong_predictions", shape=(52, 52))
