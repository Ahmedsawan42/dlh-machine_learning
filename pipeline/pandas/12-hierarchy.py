#!/usr/bin/env python3

"""a Model that rearranges two DataFrames with specific requirements"""

import pandas as pd

# Import the index function from the previous exercise
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges MultiIndex so Timestamp is the first level. Concatenates
    bitstamp and coinbase tables from timestamps 1417411980 to 1417417980.
    Adds keys to label rows from df2 as bitstamp and df1 as coinbase.
    Displays data in chronological order.

    Args:
        df1: first pd.DataFrame (coinbase)
        df2: second pd.DataFrame (bitstamp)

    Returns: the concatenated pd.DataFrame
    """
    # Index both dataframes on their Timestamp columns
    df1_indxd = index(df1)
    df2_indxd = index(df2)

    # Filter both dataframes for timestamps between 1417411980 and 1417417980.
    df1_filtered = df1_indxd[df1_indxd.index.between(1417411980, 1417417980)]
    df2_filtered = df2_indxd[df2_indxd.index.between(1417411980, 1417417980)]

    # Concatenate with keys
    concatenated = pd.concat(
        [df2_filtered, df1_filtered],
        keys=['bitstamp', 'coinbase']
    )

    # Rearrange MultiIndex so Timestamp is the first level
    # Currently: (source, timestamp) -> we want (timestamp, source)
    concatenated = concatenated.swaplevel(0, 1)

    # Sort by Timestamp (first level) in ascending order
    concatenated = concatenated.sort_index(level=0)

    return concatenated
