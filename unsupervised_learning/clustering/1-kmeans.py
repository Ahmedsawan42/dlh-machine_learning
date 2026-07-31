#!/usr/bin/env python3

"""a Model that performs K-means on a dataset"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster assignments
        Returns (None, None) on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None
        if not isinstance(k, int) or k <= 0:
            return None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None

        n, d = X.shape
        if n == 0 or d == 0:
            return None, None

        low = np.min(X, axis=0)
        high = np.max(X, axis=0)
        C = np.random.uniform(low, high, size=(k, d))

        for _ in range(iterations):
            distances = np.sqrt(
                np.sum(
                    (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            clss = np.argmin(distances, axis=1)

            sums = np.zeros((k, d))
            np.add.at(sums, (clss[:, np.newaxis], np.arange(d)), X)
            counts = np.bincount(clss, minlength=k)

            C_new = np.zeros((k, d))
            mask = counts > 0
            if np.any(mask):
                C_new[mask] = sums[mask] / counts[mask][:, np.newaxis]

            empty_mask = counts == 0
            if np.any(empty_mask):
                C_new[empty_mask] = np.random.uniform(
                    low, high, size=(np.sum(empty_mask), d)
                )

            if np.allclose(C, C_new):
                C = C_new
                break
            C = C_new

        distances = np.sqrt(
            np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
        )
        clss = np.argmin(distances, axis=1)

        return C, clss
    except Exception:
        return None, None
