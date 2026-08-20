#!/usr/bin/env python3

"""Module for saving and loading Keras model configuration in JSON format"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format

    Args:
        network: the model whose configuration should be saved
        filename: path of the file that the configuration should be saved to

    Returns:
        None
    """
    # Get the model configuration as a JSON string
    config_json = network.to_json()

    # Write the JSON string to the specified file
    with open(filename, 'w') as f:
        f.write(config_json)

    return None


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file

    Args:
        filename: path of the file containing the model's configuration
                  in JSON format

    Returns:
        The loaded model
    """
    # Read the JSON configuration from the file
    with open(filename, 'r') as f:
        config_json = f.read()

    # Create and return the model from the JSON configuration
    model = K.models.model_from_json(config_json)

    return model
