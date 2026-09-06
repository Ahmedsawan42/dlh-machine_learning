#!/usr/bin/env python3
"""
Function to create neural network layer in TensorFlow with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that includes L2 regularizat.

    Args:
        prev (tensor): Tensor containing the output of the previous layer
        n (int): Number of nodes the new layer should contain
        activation (function): The function that should be used on the layer
        lambtha (float): L2 regularization parameter

    Returns:
        tensor: Output of the new layer
    """
    # Initialize the layer's weights using VarianceScaling with fan_avg mode
    # For L2 regularization, we need to apply it to the kernel (weights)
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    # Create the regularizer
    regularizer = tf.keras.regularizers.L2(lambtha)

    # Create the Dense layer with L2 regularization on kernel weights
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        bias_regularizer=None  # Usually we don't regularize biases
    )

    # Return the output of the layer applied to prev
    return layer(prev)
