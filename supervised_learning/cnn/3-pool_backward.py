#!/usr/bin/env python3

"""Convolutional Neural Networks module."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform back propagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the output of the pooling
            layer.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
            output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the size of the kernel
            for the pooling.
        stride: tuple of (sh, sw) containing the strides for the pooling.
        mode: string containing either 'max' or 'avg', indicating whether
            to perform maximum or average pooling.

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev).
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_new, w_new, c_new = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            for k in range(c):
                if mode == 'max':
                    A_slice = A_prev[
                        :, vert_start:vert_end, horiz_start:horiz_end, k
                    ]
                    mask = A_slice == np.max(A_slice, axis=(1, 2))[
                        :, None, None
                    ]
                    dA_prev[
                        :, vert_start:vert_end, horiz_start:horiz_end, k
                    ] += mask * dA[:, i, j, k][:, None, None]
                elif mode == 'avg':
                    da = dA[:, i, j, k] / (kh * kw)
                    dA_prev[
                        :, vert_start:vert_end, horiz_start:horiz_end, k
                    ] += np.ones((kh, kw)) * da[:, None, None]

    return dA_prev
