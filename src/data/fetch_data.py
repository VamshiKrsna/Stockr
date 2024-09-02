import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, save_path="data/raw/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(os.path.join(save_path, f"{ticker}.csv"))
        print(f"Data for {ticker} saved at {save_path}")

if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL"]
    fetch_data(tickers, start_date="2015-01-01", end_date="2024-09-01")
