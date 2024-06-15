# Stockr
Stockr is an end to end ML and DL Powered App that lets you predict/forecast a stock.


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
