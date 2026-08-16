#!/usr/bin/env python3

"""Module for building a neural network model using Keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model with Keras

    Args:
        nx: number of input features
        layers: list containing number of nodes in each layer
        activations: list containing activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The Keras model
    """
    # Initialize the model using Sequential API
    model = K.Sequential()

    # Add input layer (first layer with input_shape)
    # First layer needs input_shape specified
    model.add(
        K.layers.Dense(
            units=layers[0],
            activation=activations[0],
            kernel_regularizer=K.regularizers.L2(lambtha),
            input_shape=(nx,)
        )
    )

    # Add dropout after first layer if keep_prob < 1
    if keep_prob < 1:
        model.add(K.layers.Dropout(1 - keep_prob))

    # Add remaining hidden layers and output layer
    for i in range(1, len(layers)):
        model.add(
            K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha)
            )
        )

        # Add dropout after each layer except the last one
        if i < len(layers) - 1 and keep_prob < 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
