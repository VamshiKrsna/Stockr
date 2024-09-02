import pandas as pd
import os

def preprocess_data(raw_path="data/raw/", processed_path="data/processed/"):
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    for filename in os.listdir(raw_path):
        if filename.endswith(".csv"):
            print(f"Processing {filename}...")
            data = pd.read_csv(os.path.join(raw_path, filename), index_col="Date", parse_dates=True)
            data = data.dropna()  # Drop missing values
            data['Return'] = data['Close'].pct_change()  # Example of feature engineering
            data = data.dropna()  # Drop rows with NaN values from feature engineering step
            data.to_csv(os.path.join(processed_path, filename))
            print(f"Processed data saved at {processed_path}")

if __name__ == "__main__":
    preprocess_data()
