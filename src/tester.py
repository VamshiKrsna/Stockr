import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import os
import joblib
import warnings


warnings.filterwarnings("ignore")

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


def train_lstm(data, stock):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data['Close'] = data['Close'].astype(float)

    # Split the data into train and test sets
    train_data = data[data['Date'] <= '2024-01-01']
    test_data = data[data['Date'] > '2024-01-01']

    # Debugging: Check the data split
    print(f"Training Data Size: {len(train_data)}, Testing Data Size: {len(test_data)}")
    if test_data.empty:
        print("Error: Test data is empty. Please check the date filtering logic.")
        return

    # Scaling Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
    scaled_test = scaler.transform(test_data['Close'].values.reshape(-1, 1))

    time_step = 60  # Number of previous timesteps to consider
    X_train, y_train = create_dataset(scaled_train, time_step)
    X_test, y_test = create_dataset(scaled_test, time_step)

    # Check if X_test or y_test is empty
    if X_test.size == 0 or y_test.size == 0:
        print("Error: Insufficient test data after transformation. Adjust the time step or check the test data size.")
        return

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Making Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Scaling Back to Original Values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_rescaled, test_predict[:len(y_test_rescaled)])
    print("MSE: ", mse)

    # Forecast plot with min, max, avg
    forecast_min = np.minimum(train_predict[-1] if len(train_predict) else np.zeros_like(test_predict), test_predict)
    forecast_max = np.maximum(train_predict[-1] if len(train_predict) else np.zeros_like(test_predict), test_predict)
    forecast_avg = (train_predict[-1] if len(train_predict) else np.zeros_like(test_predict) + test_predict) / 2

    forecast_min = forecast_min.reshape(-1)
    forecast_max = forecast_max.reshape(-1)
    forecast_avg = forecast_avg.reshape(-1)

    print(forecast_max)
    print(forecast_min)
    print(forecast_avg)

    plt.figure(figsize=(14, 7))
    plt.plot(train_data['Date'], train_data['Close'], label='Training', color='blue')
    plt.plot(test_data['Date'], test_data['Close'], label='Testing', color='green')
    plt.plot(test_data['Date'][:len(forecast_avg)], forecast_avg, label='Avg Forecast', color='orange')
    plt.plot(test_data['Date'][:len(forecast_min)], forecast_min, label='Min Forecast', color='red')
    plt.plot(test_data['Date'][:len(forecast_max)], forecast_max, label='Max Forecast', color='red')

    plt.fill_between(test_data['Date'][:len(forecast_avg)], forecast_min, forecast_max, color='gray', alpha=0.2)
    plt.title(f'{stock} - Actual vs Forecast (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Sample test with GOOGL.csv data
df = pd.read_csv("..//data//GOOGL.csv", index_col='Date', parse_dates=True)
train_lstm(df, "GOOGL")
