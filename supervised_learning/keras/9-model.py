#!/usr/bin/env python3

"""Module for saving and loading Keras models"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file

    Args:
        network: the model to save
        filename: path of the file that the model should be saved to

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model from a file

    Args:
        filename: path of the file that the model should be loaded from

    Returns:
        The loaded model
    """
    model = K.models.load_model(filename)
    return model
