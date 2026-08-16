#!/usr/bin/env python3

"""Module for building a neural network model using Keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model with Keras without using Sequential class

    Args:
        nx: number of input features
        layers: list containing number of nodes in each layer
        activations: list containing activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The Keras model
    """
    # Create input layer
    inputs = K.layers.Input(shape=(nx,))

    # Initialize variable to track the previous layer
    previous_layer = inputs

    # Create each layer
    for i in range(len(layers)):
        # Create dense layer with L2 regularization
        dense_layer = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha)
        )

        # Connect the dense layer to the previous layer
        current_layer = dense_layer(previous_layer)

        # Add dropout after each layer except the last one
        if i < len(layers) - 1 and keep_prob < 1:
            dropout_layer = K.layers.Dropout(1 - keep_prob)
            current_layer = dropout_layer(current_layer)

        # Update previous_layer for the next iteration
        previous_layer = current_layer

    # Create the model using the Functional API
    model = K.Model(inputs=inputs, outputs=previous_layer)

    return model
