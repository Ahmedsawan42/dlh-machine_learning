#!/usr/bin/env python3

""" This module create an Adam optimizer in TensorFlow. """

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Weight used for the first moment (0 < beta1 < 1).
        beta2 (float): Weight used for the second moment (0 < beta2 < 1).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        tensorflow.keras.optimizers.Optimizer: The Adam optimizer.
    """
    # Create Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    return optimizer
