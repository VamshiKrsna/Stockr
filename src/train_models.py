import os
import warnings
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from prophet import Prophet
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

data_dir = "../data/"
model_dir = "../models/"

os.makedirs(model_dir, exist_ok=True)



if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    for stock in stocks:
        file_path = os.path.join(data_dir, f"{stock}.csv")
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        
        print(f"Training models for {stock}...")
        
        train_arima(data, stock)
        # train_sarimax(data, stock)
        # train_lstm(data, stock)
        # train_prophet(data, stock)
        
        print(f"Models for {stock} saved successfully.")