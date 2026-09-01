#!/usr/bin/env python3

"""This module create a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tensor): Activated output of the previous layer.
        n (int): Number of nodes in the layer to be created.
        activation (callable): Activation function that should be used on the
                              output of the layer.

    Returns:
        tensor: Activated output for the layer.
    """
    # Create the Dense layer with VarianceScaling initialization
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        ),
        use_bias=False  # Bias is not used in batch normalization layer
    )

    # Apply the dense layer to previous output
    Z = dense(prev)

    # Create batch normalization layer with trainable parameters
    # The axis should be -1 (features dimension)
    gamma = tf.Variable(tf.ones((n,)), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros((n,)), trainable=True, name='beta')

    # Compute mean and variance across the batch dimension (axis=0)
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Normalize
    epsilon = 1e-7
    Z_norm = (Z - mean) / tf.sqrt(variance + epsilon)

    # Scale and shift
    Z_norm = gamma * Z_norm + beta

    # Apply activation function
    if activation is not None:
        output = activation(Z_norm)
    else:
        output = Z_norm

    return output
