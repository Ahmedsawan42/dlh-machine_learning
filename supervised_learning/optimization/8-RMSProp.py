#!/usr/bin/env python3

"""This module create an RMSProp optimizer in TensorFlow."""

import tensorflow.keras.optimizers as optim


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight (discounting factor).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        tensorflow.keras.optimizers.Optimizer: The RMSProp optimizer.
    """
    # Create RMSProp optimizer
    optimizer = optim.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)

    return optimizer
