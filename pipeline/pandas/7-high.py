#!/usr/bin/env python3

"""a Model that Sorts the DataFrame descending by the High price"""


def high(df):
    """
    Sorts the DataFrame by the High price in descending order.

    Args: df: a pd.DataFrame containing a column named High

    Returns: the sorted pd.DataFrame
    """
    return df.sort_values(by='High', ascending=False)
