# visualize.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_forecast(file_path):
    """
    Load forecast data from a CSV file.
    """
    return pd.read_csv(file_path)

def plot_forecast(stock, model_name):
    """
    Plot the forecast data.
    """
    file_path = f"forecasts/{stock}_{model_name}_forecast.csv"
    if not os.path.exists(file_path):
        st.error(f"No forecast found for {stock} using {model_name}.")
        return
    
    forecast_df = load_forecast(file_path)
    
    st.write(f"Forecast for {stock} using {model_name}")
    fig, ax = plt.subplots()
    ax.plot(forecast_df['Forecast'], label='Forecast', color='orange')
    ax.set_title(f"{stock} - {model_name} Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Stock Forecast Visualizer")

    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    models = ["LSTM", "GRU", "Transformer", "VAR", "BAR"]

    selected_stock = st.selectbox("Select Stock", stocks)
    selected_model = st.selectbox("Select Model", models)

    plot_forecast(selected_stock, selected_model)

if __name__ == "__main__":
    main()
