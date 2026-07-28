#!/usr/bin/env python3

"""a Model that performs K-means on a dataset"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate
    uniform distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None on failure
    """
    # Check for invalid inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if k > X.shape[0]:  # More clusters than data points
        return None

    # Get dimensions
    n, d = X.shape

    # Find min and max along each dimension
    min_vals = np.min(X, axis=0)  # shape (d,)
    max_vals = np.max(X, axis=0)  # shape (d,)

    # Initialize centroids with uniform distribution
    # Use numpy.random.uniform exactly once with broadcasting
    centroids = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, d)
    )

    return centroids


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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if k > X.shape[0]:
        return None, None

    n, d = X.shape

    # Initialize centroids using multivariate uniform distribution
    C = initialize(X, k)
    if C is None:
        return None, None

    # Initialize cluster assignments
    clss = np.zeros(n, dtype=int)

    # Main K-means loop - at most iterations times
    for i in range(iterations):
        # Step 1: Assign each data point to nearest centroid
        # Compute distances from each point to each centroid
        # Using broadcasting: (n, k, d) - (1, k, d) -> (n, k, d)
        # Then sum squared differences along last axis
        distances = np.sqrt(
            np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
        )

        # Assign each point to the nearest centroid
        new_clss = np.argmin(distances, axis=1)

        # Step 2: Update centroids
        new_C = np.zeros_like(C)
        cluster_counts = np.zeros(k, dtype=int)

        # Loop through clusters to compute means and check for empty clusters
        for j in range(k):
            # Get points in cluster j
            mask = (new_clss == j)
            cluster_points = X[mask]
            count = np.sum(mask)
            cluster_counts[j] = count

            if count > 0:
                # Compute mean of points in cluster
                new_C[j] = np.mean(cluster_points, axis=0)
            else:
                # Empty clst- reinitialize centroid using uniform distribution
                min_vals = np.min(X, axis=0)
                max_vals = np.max(X, axis=0)
                new_C[j] = np.random.uniform(
                    low=min_vals,
                    high=max_vals,
                    size=(1, d)
                )
                # Keep the cluster count as 0 for this centroid
                cluster_counts[j] = 0

        # Check for convergence (no change in centroids)
        if np.allclose(C, new_C):
            # After convergence, ensure all clusters have at least one point
            # If not, we need to handle this case
            if np.all(cluster_counts > 0):
                return C, clss
            else:
                # Some clusters are empty after convergence, continue
                pass

        # Update centroids and assignments
        C = new_C
        clss = new_clss

        # Check if all clusters have at least one point
        # If a cluster is empty, we continue to next iteration
        if np.any(cluster_counts == 0):
            continue

    # Final assignment
    distances = np.sqrt(
        np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
    )
    clss = np.argmin(distances, axis=1)

    return C, clss
