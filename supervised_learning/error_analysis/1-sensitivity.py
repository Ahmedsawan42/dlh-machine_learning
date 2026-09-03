#!/usr/bin/env python3

"""This module calculates the sensitivity confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (true positive rate) for each class.

    Sensitivity for a class is defined as: TP / (TP + FN)
    where TP = true positives for the class
          FN = false negatives for the class

    In the confusion matrix:
    - True positives for class i are at position [i, i]
    - False negatives for class i are the sum of row i excluding [i, i]

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where
                   row indices represent correct labels and column indices
                   represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
        of each class. Returns 0 for classes with no positive samples
        to avoid division by zero.

    Raises:
        ValueError: If confusion matrix is not square
    """
    # Check if confusion matrix is square
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")

    classes = confusion.shape[0]
    sensitivities = np.zeros(classes, dtype=np.float64)

    for i in range(classes):
        # True positives: diagonal element for class i
        tp = confusion[i, i]

        # False negatives: sum of row i minus the diagonal element
        # (samples of class i incorrectly predicted as other classes)
        fn = np.sum(confusion[i, :]) - tp

        # Calculate sensitivity, handling division by zero
        if tp + fn > 0:
            sensitivities[i] = tp / (tp + fn)
        else:
            # No samples of this class in the dataset
            sensitivities[i] = 0.0

    return sensitivities
