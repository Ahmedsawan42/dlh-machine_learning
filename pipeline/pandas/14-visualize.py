#!/usr/bin/env python3

"""a Model that visualize the DataFrame"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'], errors='ignore')

# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df = df.set_index('Date')

# Missing values in Close should be set to the previous row value
df['Close'] = df['Close'].fillna(method='ffill')

# Missing values in High, Low, Open should be set to the same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 and beyond
df_2017 = df[df.index >= '2017-01-01']

# Resample at daily intervals and aggregate
daily_df = df_2017.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(12, 8))

# Plot Close price
plt.subplot(2, 1, 1)
plt.plot(daily_df.index, daily_df['Close'], label='Close Price', color='blue')
plt.title('Daily Close Price (2017 and beyond)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot Volume
plt.subplot(2, 1, 2)
plt.plot(
        daily_df.index,
        daily_df['Volume_(BTC)'],
        label='Volume (BTC)',
        color='orange'
    )
plt.title('Daily Volume (BTC) - 2017 and beyond')
plt.xlabel('Date')
plt.ylabel('Volume (BTC)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Return the transformed pd.DataFrame before plotting
transformed_df = daily_df
