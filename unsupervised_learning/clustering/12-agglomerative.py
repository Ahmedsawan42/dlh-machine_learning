#!/usr/bin/env python3

"""a Model that performs agglomerative clustering on a dataset"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform agglomerative clustering with Ward linkage and display dendrogram.
    """
    try:
        if not hasattr(X, 'shape') or len(X.shape) != 2:
            return None
        n, d = X.shape
        if n == 0 or d == 0:
            return None
        if not isinstance(dist, (int, float)) or dist <= 0:
            return None

        Z = scipy.cluster.hierarchy.linkage(X, method='ward')
        clss = scipy.cluster.hierarchy.fcluster(
            Z, t=dist, criterion='distance'
        ) - 1

        plt.figure()
        scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
        plt.show()

        return clss
    except Exception:
        return None
