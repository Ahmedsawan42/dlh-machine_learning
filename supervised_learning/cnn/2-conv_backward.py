#!/usr/bin/env python3

"""Convolutional Neural Networks module."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform back propagation over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the unactivated output of the
            convolutional layer.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution.
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution.
        padding: string that is either 'same' or 'valid', indicating the
            type of padding used.
        stride: tuple of (sh, sw) containing the strides for the convolution.

    Returns:
        dA_prev: partial derivatives with respect to the previous layer.
        dW: partial derivatives with respect to the kernels.
        db: partial derivatives with respect to the biases.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev_padded[
                :, vert_start:vert_end, horiz_start:horiz_end, :
            ]

            for k in range(c_new):
                dA_prev_padded[
                    :, vert_start:vert_end, horiz_start:horiz_end, :
                ] += dZ[:, i, j, k][:, None, None, None] * W[:, :, :, k]

                dW[:, :, :, k] += np.sum(
                    A_slice * dZ[:, i, j, k][:, None, None, None],
                    axis=0
                )

                db[:, :, :, k] += np.sum(dZ[:, i, j, k])

    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
