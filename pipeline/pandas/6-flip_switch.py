#!/usr/bin/env python3

"""a Model that takes df and Returns the transformed sorted df"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes it.

    Args:
        df: a pd.DataFrame with a datetime index or timestamp column

    Returns: the transformed pd.DataFrame
    """
    # Sort the DataFrame in reverse chronological order
    # Assuming the index is a datetime index
    sorted_df = df.sort_index(ascending=False)

    # Transpose the sorted DataFrame
    transposed = sorted_df.T

    return transposed
