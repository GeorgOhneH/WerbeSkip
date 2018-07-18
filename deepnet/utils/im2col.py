# code from http://cs231n.github.io/convolutional-networks/

import numpywrapper as np


def get_im2col_indices(x_shape, field_height, field_width, padding_h=1, padding_w=1, stride=1):
    # First figure out what the size of the output should be
    N, C, W, H = x_shape
    assert (H + 2 * padding_h - field_height) % stride == 0
    assert (W + 2 * padding_w - field_width) % stride == 0
    out_height = int((H + 2 * padding_h - field_height) / stride + 1)
    out_width = int((W + 2 * padding_w - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding_h=1, padding_w=1,  stride=1, padding_value=0):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding_w, padding_w), (padding_h, padding_h)), mode='constant', constant_values=padding_value)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding_h, padding_w,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding_h=1, padding_w=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, W, H = x_shape
    H_padded, W_padded = H + 2 * padding_h, W + 2 * padding_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding_h, padding_w,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    # np.scatter_add is in numpy np.add.at
    np.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding_h == 0 and padding_w == 0:
        return x_padded
    return x_padded[:, :, padding_w:-padding_w, padding_h:-padding_h]
