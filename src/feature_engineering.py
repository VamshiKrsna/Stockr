# In this script, We will be further add and improve the Dataset by adding more features like Rolling Window features.
import pandas as pd
import datetime
from datetime import *

def add_features(df):

    # Lets drop Dividends and Stock Splits as they are not very effective and useful for forecast
    df = df.drop(["Dividends","Stock Splits"],axis = 1)

    #Lets extract day, day of week, etc.
    # print(df['Date'].dtype)
    # df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # print(df['Date'].dtype)

    # df['Date'] = df['Date'].dt.date
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z").date())
    # df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S%z").dt.date

    # Add rolling window features
    # Adding Rolling mean features for 3,9,18 day frequencies.
    df['S_3'] = df['Close'].rolling(window=3).mean()
    df['S_9'] = df['Close'].rolling(window=9).mean()
    df['S_18'] = df['Close'].rolling(window=18).mean()
    
    # Lag Features :
    for i in range(1, 4):
        df[f'lag_{i}'] = df['Close'].shift(i)

    # Rolling Window Features : 
    df['Rolling_Mean'] = df['Close'].rolling(window=3).mean()
    df['Rolling_Min'] = df['Close'].rolling(window=3).min()
    df['Rolling_Max'] = df['Close'].rolling(window=3).max()

    # Exponential Moving Average
    df["EMA_12"] = df['Close'].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df['Close'].ewm(span=12, adjust=False).mean()

    # MACD = Moving Average Convergence Divergence
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI = Relative Strength Index
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # Overall statistics
    df['Overall_Mean'] = df['Close'].mean()
    df['Overall_Min'] = df['Close'].min()
    df['Overall_Max'] = df['Close'].max()

    # Add target (Close) variable
    df['Close'] = df['Close'].shift(-1)
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    for stock in ["AAPL","MSFT","GOOGL","AMZN","TSLA"]:
        file_path = f"../data/{stock}.csv"
        df = pd.read_csv(file_path)
        df = add_features(df)
        df.to_csv(file_path, index=False)
        print(f"Features added for {stock}")
        # print(df.head(5))