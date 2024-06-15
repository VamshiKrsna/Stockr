import yfinance as yf
import pandas as pd
import pandas_datareader as pdr

def fetch_stock_data(stock_symbol, start_date):
    ticker = yf.Ticker(stock_symbol)
    df = ticker.history(start=start_date)
    return df

def save_to_csv(stock_symbol, df):
    file_path = f"../data/{stock_symbol}.csv"
    df.to_csv(file_path)

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2006-01-01" # Lets try for 18 years of data
    # All stocks except TSLA have data from 2006, due to late launch of TSLA's IPO we have data from 2010  
    
    for stock in stocks:
        df = fetch_stock_data(stock, start_date)
        save_to_csv(stock, df)
        print(f"Data for {stock} saved.")