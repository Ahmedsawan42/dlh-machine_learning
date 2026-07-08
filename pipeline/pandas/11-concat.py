#!/usr/bin/env python3

"""a Model that concatenates two DataFrames with specific requirements"""

import pandas as pd

# Import the index function from the previous exercise
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames with specific requirements.

    Args:
        df1: first pd.DataFrame (coinbase)
        df2: second pd.DataFrame (bitstamp)

    Returns: the concatenated pd.DataFrame
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Includes all timestamps from df2 to and including timestamp 1417411920.
    df2_filtered = df2_indexed[df2_indexed.index <= 1417411920]

    # Concatenate: df2 rows (bitstamp) on top of df1 rows (coinbase)
    # Add keys to label the data sources
    concatenated = pd.concat(
        [df2_filtered, df1_indexed],
        keys=['bitstamp', 'coinbase']
    )

    return concatenated
