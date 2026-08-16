#!/usr/bin/env python3

"""Module for converting label vectors to one-hot matrices"""

import numpy as np


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
        classes = np.max(labels) + 1

    # Create one-hot matrix using NumPy
    one_hot_matrix = np.eye(classes)[labels]

    return one_hot_matrix
