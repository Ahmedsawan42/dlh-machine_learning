#!/usr/bin/env python3

"""a Model that performs K-means on a dataset"""

import sklearn.cluster


def kmeans(X, k):
    """Perform K-means clustering on dataset X with k clusters."""
    try:
        if not hasattr(X, 'shape') or len(X.shape) != 2:
            return None, None
        if not isinstance(k, int) or k <= 0:
            return None, None

        n, d = X.shape
        if n == 0 or d == 0:
            return None, None

        model = sklearn.cluster.KMeans(n_clusters=k)
        model.fit(X)

        C = model.cluster_centers_
        clss = model.labels_

        return C, clss
    except Exception:
        return None, None
