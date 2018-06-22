from deepnet import Network
from app.image_processing.image_loader import load_ads_cnn
from app.image_processing.generator import TrainGenerator
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam


if __name__ == "__main__":
    generator = TrainGenerator(epochs=1, mini_batch_size=128, padding=10, n_workers=1)

    validation_data, validation_labels, test_data, test_labels = load_ads_cnn(split=0.8, padding=10)

    net = Network()

    net.use_gpu = True

    net.input((3, 52, 52))

    net.add(ConvolutionLayer(n_filter=32, width_filter=4, height_filter=4, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
    net.add(ConvolutionLayer(n_filter=64, width_filter=4, height_filter=4, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(128))
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.03)
    net.regression(optimizer=optimizer, cost="cross_entropy")

    net.fit_generator(generator, validation_set=(validation_data, validation_labels), snapshot_step=0.5, save_step=100)
    net.evaluate(test_data, test_labels)
    net.save_wrong_predictions(test_data, test_labels, "wrong_predictions", shape=(3, 52, 52))
