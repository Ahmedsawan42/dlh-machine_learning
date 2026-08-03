#!/usr/bin/env python3

"""a Model that calculates a GMM from a datase"""

import sklearn.mixture


def gmm(X, k):
    """Calculate a GMM from a dataset."""
    try:
        if not isinstance(k, int) or k < 1:
            return None, None, None, None, None

        n, d = X.shape
        if n == 0 or d == 0:
            return None, None, None, None, None
        if k > n:
            return None, None, None, None, None

        model = sklearn.mixture.GaussianMixture(n_components=k)
        model.fit(X)

        pi = model.weights_
        m = model.means_
        S = model.covariances_
        clss = model.predict(X)
        bic = model.bic(X)

        return pi, m, S, clss, bic
    except Exception:
        return None, None, None, None, None
