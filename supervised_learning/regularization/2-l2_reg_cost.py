#!/usr/bin/env python3

"""
Function to calculate the cost of a neural network with L2 regularization
using a Keras model.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tensor): Tensor containing the cost of the network without L2reg
        model (Keras model): Keras model that includes layers with L2 regularz

    Returns:
        tensor: Tensor containing the total cost for each layer of the network
                accounting for L2 regularization
    """
    # List to store the cost for each layer
    layer_costs = []

    # Iterate through all layers in the model
    for layer in model.layers:
        # Check if the layer has L2 regularization losses
        if hasattr(layer, 'losses') and layer.losses:
            # For each layer, add its regularization loss to the base cost
            # Sum the losses for this layer and add to the base cost
            layer_regularization = tf.reduce_sum(layer.losses)
            layer_costs.append(cost + layer_regularization)

    # Stack the costs into a tensor
    # Note: The first element is the base cost without regularization
    # The subsequent elements are the base cost +regularization for each layer
    return tf.stack(layer_costs)
