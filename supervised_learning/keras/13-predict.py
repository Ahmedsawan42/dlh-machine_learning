#!/usr/bin/env python3

"""Module for making predictions with a Keras model"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network model

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: boolean that determines if output should be printed
                 during the prediction process

    Returns:
        The prediction for the data
    """
    # Make predictions using the model
    predictions = network.predict(
        x=data,
        verbose=verbose
    )

    return predictions
