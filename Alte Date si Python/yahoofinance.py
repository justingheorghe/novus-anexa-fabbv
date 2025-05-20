import yfinance as yf

# Define the ticker symbol
ticker = 'ETH-USD'

# Get data from May 10, 2022 to May 18, 2025
data = yf.download(ticker, start='2020-03-10')

# Save to CSV
data.to_csv('eth-usd_3y_history.csv')
