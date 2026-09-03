#!/usr/bin/env python3

"""This module calculates the specificity for Confusion matrix."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity (true negative rate) for each class.

    Specificity for a class is defined as: TN / (TN + FP)
    where TN = true negatives for the class
          FP = false positives for the class

    In the confusion matrix:
    - True negatives for class i = sum of all elements not in rowi or column i
    - False positives for class i = sum of column i excluding [i, i]

    Alternatively, specificity can be calculated as:
    (Total samples - (TP + FN + FP)) / (Total samples - (TP + FN))

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where
                   row indices represent correct labels and column indices
                   represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class. Returns 0 for classes with no negative samples
        to avoid division by zero.

    Raises:
        ValueError: If confusion matrix is not square
    """
    # Check if confusion matrix is square
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")

    classes = confusion.shape[0]
    total_samples = np.sum(confusion)
    specificities = np.zeros(classes, dtype=np.float64)

    for i in range(classes):
        # True positives for class i (diagonal element)
        tp = confusion[i, i]

        # False negatives: sum of row i minus diagonal
        fn = np.sum(confusion[i, :]) - tp

        # False positives: sum of column i minus diagonal
        fp = np.sum(confusion[:, i]) - tp

        # True negatives: total samples minus all elements in rowi and columni
        tn = total_samples - (tp + fn + fp)

        # Calculate specificity, handling division by zero
        if tn + fp > 0:
            specificities[i] = tn / (tn + fp)
        else:
            # No negative samples for this class
            specificities[i] = 0.0

    return specificities
