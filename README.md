# Stockr
Stockr is an end to end ML and DL Powered App that lets you predict/forecast a stock.

<p align="center">
    <img width="200" src="src/stockr_logo.png" alt="Stockr Logo">
</p>

**THIS PROJECT IS UNDER DEVELOPMENT**

### How Does **Stockr** Work ? 


1. Fetches Data from yfinance API
2. Preprocess and Scales Data.
3. Feature Engineering : Appends additional useful indicators and features to the data.
4. Model training : Trains Models like ARIMA, SARIMAX, LSTM, Prophet, etc. on the stock data.
5. Forecast & Predict : Makes forecast and predictions on the historical engineered data.
6. Reiterate : Retrains and improves on further iterations.


### 1. Data Collection 


Stockr Uses yfinance api to extract stock data. Currently, Stockr supports Apple, Microsoft, Amazon, Google, Tesla Stocks data.


### 2. Data Preprocessing


Collected data is then cleaned, preprocessed, scaled using Scikit Learn's train test split and Standard Scaler.


### 3. Feature Engineering


- We then extract more meaningful features and Indicators from the preprocessed data.
- We can simply add features like day of week, quarter, Month, etc.
- These Features Include : Rolling Window Features, Lag Features, Exponential Moving Average, Moving Average Convergence Divergence, Relative Strength Index, Bollinger Bands, Volume Based Features.


  #### How do these new features enhance forecasting and predictions ?


  #### **1. Lag Features (Lag_1, Lag_2):**

  What it captures: Past stock prices.
  How it helps: Provides context for recent price trends and helps the model recognize short-      term momentum or reversal patterns.


  #### **2. Rolling Window Features (Rolling_Mean, Rolling_Min, Rolling_Max, S_3, S_9, S_18):**

    What it captures: Moving averages and extremes over different time windows.
    How it helps: Identifies trends and volatility over short to medium time frames, smoothing       out noise and highlighting significant movements.


  #### **3. Exponential Moving Averages (EMA_12, EMA_26):**

    What it captures: Weighted moving averages that give more importance to recent prices.
    How it helps: Emphasizes recent price action, which can be more relevant for predicting       near-term future prices.


  #### **4. MACD and MACD Signal:**

    What it captures: Difference between two EMAs and its signal line.
    How it helps: Detects changes in the strength, direction, momentum, and duration of a trend.     Commonly used to generate buy/sell signals.


  #### **5. Relative Strength Index (RSI):**

    What it captures: Magnitude of recent price changes.
    How it helps: Indicates overbought or oversold conditions, helping to predict potential       reversals or continuations of a trend.


  #### **6. Overall Statistics (Overall_Mean, Overall_Min, Overall_Max):**

    What it captures: Long-term averages and extremes.
    How it helps: Provides a baseline or reference for the stock's performance over the entire dataset.
