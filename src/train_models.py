import os
import warnings
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from prophet import Prophet
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ARIMA, SARIMAX, etc are too basic models for our usecase.
# Considering Prophet, Monte Carlo Simulations, LSTM etc.
# 1. Monte Carlo
# 2. LSTM
# 3. Prophet
# 4. Markov Chains


warnings.filterwarnings("ignore")

data_dir = "../data/"
model_dir = "../models/"

os.makedirs(model_dir, exist_ok=True)

def train_lstm(df, target_col_index=3, forecast_days=30):
    """
    Trains an LSTM model on the given data and forecasts the next `forecast_days` days.

    Parameters:
        df (pd.DataFrame): The input dataframe with datetime index and numeric columns.
        target_col_index (int): The index of the target column ('Close' price typically).
        forecast_days (int): The number of days to forecast. Default is 30.

    Returns:
        None: Displays the plot of actual vs forecasted values and prints the MSE.
    """
    
    # Selecting numeric and datetime columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[numeric_cols])

    # Splitting the data into training and testing sets
    train_data, test_data = scaled_data[:-forecast_days], scaled_data[-(forecast_days + 60):]  # Take the last 60 days + forecast days for testing

    # Function to create input (X) and output (y) datasets
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, target_col_index])  # Using the target column index
        return np.array(X), np.array(y)

    time_step = 60  # Number of time steps to consider for each sample
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Check if X_test and y_test are non-empty and have the correct shape before reshaping
    if X_test.size == 0 or y_test.size == 0:
        print("Error: Not enough data points for testing. Adjust the forecast period or time step.")
        return
    
    # Reshape the data for the LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Forecast the next `forecast_days` using the last available data
    test_predict = model.predict(X_test)

    # Scale back to original values
    test_predict_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((test_predict.shape[0], X_test.shape[2] - 1)), test_predict)))
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((y_test.shape[0], X_test.shape[2] - 1)), y_test.reshape(-1, 1))))

    # Evaluate the model
    mse = mean_squared_error(y_test_rescaled[:, -1], test_predict_rescaled[:, -1])
    print('MSE: ', mse)

    # Plot actual vs forecasted values
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, -1], label='Actual', color='blue')
    plt.plot(range(len(test_predict_rescaled)), test_predict_rescaled[:, -1], label='Forecasted', color='orange')
    plt.title(f'Actual vs Forecasted Prices for the Last {forecast_days} Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


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
    model_path = os.path.join('models', f"{stock}_sarimax.joblib")
    joblib.dump(model_fit, model_path)

def train_markov(data, stock):
    pass

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    for stock in stocks:
        file_path = os.path.join(data_dir, f"{stock}.csv")
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        print(f"Training models for {stock}...")

        # train_prophet(data, stock)
        # train_sarimax(data, stock)
        train_lstm(data, stock)
        # train_prophet(data, stock)

        print(f"Models for {stock} saved successfully.")









