#!/usr/bin/env python3

"""This module create a momentum optimizer in TensorFlow."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets upgradient descent with momentum optimization algorithm in TensorFlow

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight (0 < beta1 < 1).

    Returns:
        tensorflow.keras.optimizers.Optimizer: The momentum optimizer.
    """
    # Create SGD optimizer with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
