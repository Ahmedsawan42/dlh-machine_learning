#!/usr/bin/env python3

"""a Model that takes a pd.DataFrame and returns sliced pd.DataFrame"""


def slice(df):
    """
    Extracts the columns High, Low, Close, and Volume_(BTC)
    and selects every 60th row from these columns.

    Args: df: a pd.DataFrame

    Returns: the sliced pd.DataFrame
    """
    # Extract and slice in one line
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
