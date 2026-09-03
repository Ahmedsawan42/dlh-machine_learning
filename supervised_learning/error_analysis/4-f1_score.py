#!/usr/bin/env python3

"""This module calculates the F1 score for confusion matrices."""

import numpy as np

# Import the previously created functions
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class from a confusion matrix.

    The F1 score is the harmonic mean of precision and sensitivity (recall):
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    The F1 score ranges from 0 to 1, with 1 being the best possible score.

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where
                   row indices represent correct labels and column indices
                   represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score
        of each class. Returns 0 for classes where both precision and
        sensitivity are 0 to avoid division by zero.

    Raises:
        ValueError: If confusion matrix is not square
    """
    # Check if confusion matrix is square
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")

    # Get precision and sensitivity for each class
    prec = precision(confusion)
    sens = sensitivity(confusion)

    classes = confusion.shape[0]
    f1_scores = np.zeros(classes, dtype=np.float64)

    for i in range(classes):
        # Calculate F1 score
        if prec[i] + sens[i] > 0:
            f1_scores[i] = 2 * (prec[i] * sens[i]) / (prec[i] + sens[i])
        else:
            # Both precision and sensitivity are 0
            f1_scores[i] = 0.0

    return f1_scores
