# src/data/fetch_data.py
import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, save_path="data/raw/"):
    """
    Fetches stock data from Yahoo Finance and saves it to CSV files.
    
    Parameters:
    - tickers: List of stock symbols.
    - start_date: Start date for data fetching.
    - end_date: End date for data fetching.
    - save_path: Directory where the raw data will be saved.
    """
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
