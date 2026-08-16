#!/usr/bin/env python3

"""Module for converting label vectors to one-hot matrices"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix

    Args:
        labels: label vector to convert
        classes: number of classes (optional)

    Returns:
        The one-hot matrix
    """
    # If classes is not provided, determine it from the labels
    if classes is None:
        classes = int(K.backend.max(labels) + 1)

    # Create one-hot matrix using Keras utilities
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot_matrix
