import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, GRU
from sklearn.model_selection import train_test_split

def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_gru_model(data, epochs=50, batch_size=32):
    X_train, y_train = data
    model = create_gru_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

if __name__ == "__main__":
    df = pd.read_csv('../data/processed/AAPL.csv')
    data = df['Scaled_Close'].values
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_gru_model((X, y))
