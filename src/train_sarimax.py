import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

data_dir = "../data/"
model_dir = "../models/"


def train_sarimax(data, stock):
    data['Date'] = pd.to_datetime(data.index).tz_localize(None)
    data['Close'] = data['Close'].astype(float)
    
    train_data = data[data['Date'] <= '2023-01-01']
    test_data = data[data['Date'] > '2023-01-01']
    
    # SARIMAX Model
    model = sm.tsa.statespace.SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    
    # Forecasting
    forecast = model_fit.get_forecast(steps=len(test_data))
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_min = conf_int.iloc[:, 0]
    forecast_max = conf_int.iloc[:, 1]
    
    mse = mean_squared_error(test_data['Close'], forecast_mean)
    print("MSE:", mse)
    
    # Plotting the Results
    plt.figure(figsize=(14, 7))
    plt.plot(train_data['Date'], train_data['Close'], label='Training', color='blue')
    plt.plot(test_data['Date'], test_data['Close'], label='Testing', color='green')
    plt.plot(test_data['Date'], forecast_mean, label='Forecast Avg', color='orange')
    plt.fill_between(test_data['Date'], forecast_min, forecast_max, color='gray', alpha=0.2, label='Min-Max Forecast')
    plt.title(f'{stock} - Actual vs Forecast (SARIMAX)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Saving the Model
    model_path = os.path.join(model_dir, f"{stock}_sarimax.joblib")
    joblib.dump(model_fit, model_path)
    print(f"Model for {stock} saved successfully.")

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    for stock in stocks:
        file_path = os.path.join(data_dir, f"{stock}.csv")
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        print(f"Training models for {stock}...")

        # Train SARIMAX models for each stock and save them in the models directory
        train_sarimax(data, stock)

        print(f"Models for {stock} saved successfully.")