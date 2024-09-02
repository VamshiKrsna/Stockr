# This file will single handedly fetch, preprocess and engineer features.
import os
# First fetch data : 
os.system("python fetch_data.py")

# Second preprocess data : 
os.system("python data_preprocessing.py")

# Third Feature Engineering : 
os.system("python feature_engineering.py")
