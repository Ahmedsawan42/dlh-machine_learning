#!/usr/bin/env python3

"""a Model that Fills missing different values"""


def fill(df):
    """
    Removes the Weighted_Price column.
    Fills missing values in Close with previous row's value.
    Fills missing values in High, Low, Open with corresponding Close value.
    Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.

    Args: df: a pd.DataFrame

    Returns: the modified pd.DataFrame
    """
    # Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill missing values in Close with previous row's value (forward fill)
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill missing values in High, Low, Open with corresponding Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Sets missing values in Volume_(BTC) and Volume_(Currency) to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
