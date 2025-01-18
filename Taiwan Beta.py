# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:01:20 2025

@author: user
"""

#Package
import numpy as np
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

# Create directory if it doesn't exist
file_path = r'D:\Taiwan Research'
if not os.path.exists(file_path):
    os.makedirs(file_path)
    print(f"Created directory: {file_path}")

# List of symbols
symbols = [
    '0050.TW', '0056.TW', '00662.TW', '00701.TW', '00702.TW', 
    '00703.TW', '00757.TW', '00830.TW', '00850.TW', '00878.TW',
    '00881.TW', '00882.TW', '00891.TW', '00893.TW', '00894.TW',
    '00915.TW', '00918.TW', '00919.TW', '00927.TW', '00929.TW',
    '00934.TW', '00936.TW', '00939.TW', '00940.TW', '00944.TW',
    '00946.TW', '00954.TW', '2330.TW', '^TWII'
]

# Set date range
end_date = datetime.now()
start_date = datetime(2024, 7, 1)  # Start from July 1, 2024

# Download data
all_data = {}
failed_downloads = []
successful_downloads = []

# Download adjusted close prices only
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
        if len(data) > 0:
            all_data[symbol] = data
            successful_downloads.append(symbol)
            print(f"Successfully downloaded {symbol}")
        else:
            failed_downloads.append(symbol)
            print(f"No data found for {symbol}")
    except Exception as e:
        failed_downloads.append(symbol)
        print(f"Failed to download {symbol}: {str(e)}")

# Create DataFrame from all successful downloads
df = pd.DataFrame(all_data)

# Save to Excel
output_file = os.path.join(file_path, 'taiwan_etf_stocks_data.xlsx')
df.to_excel(output_file)

print("\nDownload complete!")
print(f"Data saved to: {output_file}")

# Print basic information
if len(df) > 0:
    print("\nDataset Information:")
    print(f"Date Range: {df.index[0]} to {df.index[-1]}")
    print(f"Number of trading days: {len(df)}")
    print(f"Number of securities: {len(df.columns)}")

    # Check for missing data
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("\nMissing data points per security:")
        print(missing_data[missing_data > 0])

if failed_downloads:
    print("\nFailed downloads:")
    for symbol in failed_downloads:
        print(symbol)