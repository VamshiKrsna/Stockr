from prophet import Prophet
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

def train_prophet(data, stock):
    """
    prophet has a stan_backend error (tried so hard, can't resolve yet)
    """
    temp = data
    temp["Date"] = temp.index
    temp['Date'] = temp['Date'].dt.tz_localize(None)
    prop_data = temp.copy()
    prop_data['ds'] = pd.to_datetime(prop_data.index)
    prop_data['ds'] = prop_data['ds'].dt.tz_localize(None)
    prop_data['y'] = prop_data['Close']
    prop_data = prop_data.rename(columns={'ds': 'ds', 'y': 'y'})

    train_data = prop_data[prop_data['ds'] <= '2023-01-01']
    test_data = prop_data[prop_data['ds'] > '2023-01-01']

    model = Prophet()
    model.fit(train_data)

    last_date = train_data['ds'].max()
    until_date = pd.to_datetime('2024-07-31')
    periods = (until_date - last_date).days
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    forecast_test = forecast[forecast['ds'] >= test_data['ds'].min()]

    mse = mean_squared_error(test_data['y'], forecast_test['yhat'][:len(test_data)])
    print("MSE : ", mse)

    """
    For Visualizing difference between forecast and actual values, use the following code.
    Inorder to continue training all other stocks, you must close every plot.  
    """ 
    fig, ax = plt.subplots()
    ax.plot(train_data['ds'], train_data['y'], label='Training', color='blue')
    ax.plot(test_data['ds'], test_data['y'], label='Testing', color='green')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='orange')
    ax.set_title('Actual vs Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()

    # Saving the model
    model_path = os.path.join(model_dir, f"{stock}_prophet.joblib")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    for stock in stocks:
        file_path = os.path.join(data_dir, f"{stock}.csv")
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        print(f"Training models for {stock}...")

        # Train LSTM models for each stock and save them in the models directory
        train_prophet(data, stock_name=stock, forecast_days=30, epochs=20)

        print(f"Models for {stock} saved successfully.")