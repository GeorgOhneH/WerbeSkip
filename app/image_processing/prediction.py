from deepnet.utils.im2col import im2col_indices
from deepnet.utils import make_batches
import numpy as np
import cupy
import cv2
from tqdm import tqdm


def visualize_prediction(network, images, logo_w, logo_h, stride=1):
    size, depth, width, height = images.shape

    padding_w = (logo_w-1)//2
    padding_h = (logo_h-1)//2

    out_height = int((height + 2 * padding_h - logo_h) / stride + 1)
    out_width = int((width + 2 * padding_w - logo_w) / stride + 1)

    results = []
    batches = make_batches(images, 1)
    for batch in tqdm(batches):
        network.use_gpu = False
        batch_size = batch.shape[0]
        crop_images = im2col_indices(batch, field_height=logo_h, field_width=logo_w, padding_w=padding_w,
                                     padding_h=padding_h, stride=stride, padding_value=1).transpose().reshape(-1, depth, logo_w, logo_h)
        rearranged_crop_images = np.concatenate([crop_images[x::batch_size] for x in range(batch_size)])
        network.use_gpu = True
        results.append(cupy.asnumpy(network.feedforward(rearranged_crop_images)))
    network.use_gpu = False
    y = np.concatenate(results)

    confidence = y[:, 0].reshape(size, out_width, out_height)

    imgs = np.zeros((size, out_width, out_height, 3), dtype="float32")
    imgs[..., 2] = 200  # hue | max = 255
    imgs[..., 1] = abs(confidence - 0.5) * 2  # saturation | max = 1
    imgs[confidence < 0.5, 0] = 240  # set colour to blue | default is red | max = 358
    for index, (img, orig, conf) in enumerate(zip(imgs, images, confidence)):
        orig.dtype = "float32"
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imwrite("colourmap{}.png".format(index), img)
        cv2.imwrite("gray{}.png".format(index), conf*255)
        cv2.imwrite("orig{}.png".format(index), orig.transpose(1, 2, 0) * 255)
