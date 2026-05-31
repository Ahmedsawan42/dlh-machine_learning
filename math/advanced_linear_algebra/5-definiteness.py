#!/usr/bin/env python3

"""
Module for matrix definiteness calculation.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a square matrix using Sylvester's criterion.

    Args:
        matrix: A numpy.ndarray of shape (n, n)

    Returns:
        String: "matrix definiteness"
        or None: If matrix is invalid or doesn't fit any category

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """

    # Check the matrix.
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return None
    # Check if matrix is symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None

    n = matrix.shape[0]
    tolerance = 1e-10

    # Calculate determinants of all leading principal minors
    leading_minors = []
    for k in range(1, n + 1):
        submatrix = matrix[:k, :k]
        det = np.linalg.det(submatrix)
        leading_minors.append(det)

    # Check signs of all leading principal minors
    all_positive = all(det > tolerance for det in leading_minors)
    all_negative_alternating = True

    # For negative definite: 1st minor < 0, 2nd minor > 0, 3rd minor < 0, ...
    # So sign alternates starting with negative
    for i, det in enumerate(leading_minors):
        expected_sign = (-1) ** (i + 1)  # (-1)^(k) where k is 1-indexed
        if expected_sign == 1:  # Expect positive
            if det <= tolerance:
                all_negative_alternating = False
                break
        else:  # Expect negative
            if det >= -tolerance:
                all_negative_alternating = False
                break

    # Check for semi-definite cases (some minors may be zero)
    # Check all principal minors (not just leading ones) for semi-definite
    # For simplicity, we'll use eigenvalue approach for semi-definite cases
    if all_positive:
        return "Positive definite"
    elif all_negative_alternating:
        return "Negative definite"
    else:
        # For semi-definite and indefinite, use eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)

        has_positive = np.any(eigenvalues > tolerance)
        has_negative = np.any(eigenvalues < -tolerance)
        has_zero = np.any(np.abs(eigenvalues) < tolerance)

        if has_positive and not has_negative and has_zero:
            return "Positive semi-definite"
        elif has_negative and not has_positive and has_zero:
            return "Negative semi-definite"
        elif has_positive and has_negative:
            return "Indefinite"
        else:
            return None
