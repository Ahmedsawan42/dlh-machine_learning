#!/usr/bin/env python3

"""Convolutional Neural Networks module."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Perform forward propagation over a convolutional layer of a neural network

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution.
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution.
        activation: activation function applied to the convolution.
        padding: string that is either 'same' or 'valid', indicating the
            type of padding used.
        stride: tuple of (sh, sw) containing the strides for the convolution.

    Returns:
        The output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = 0
        pw = 0

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                A_slice = A_prev_padded[
                    :, vert_start:vert_end, horiz_start:horiz_end, :
                ]
                Z[:, i, j, k] = np.sum(
                    A_slice * W[:, :, :, k], axis=(1, 2, 3)
                ) + b[:, :, :, k]

    return activation(Z)
