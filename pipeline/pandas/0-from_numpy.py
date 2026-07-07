#!/usr/bin/env python3

"""a Model to creates a pd.DataFrame from a np.ndarray"""

import pandas as pd


def from_numpy(array):
    """
    creates a pd.DataFrame from a np.ndarray.

    Args: array (a np.ndarray).

    Returns: pd a pd.DataFrame.
    """
    # Get the number of columns
    num_cols = array.shape[1]

    # Generate column labels: A, B, C, ... up to the number of columns
    columns = [chr(65 + i) for i in range(num_cols)]

    # Create DataFrame with the specified column labels
    df = pd.DataFrame(array, columns=columns)

    return df
