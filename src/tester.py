import os
import sys

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the scripts relative to the base directory
data_preprocessing_path = os.path.join(base_dir, 'src', 'data', 'data_preprocessing.py')
feature_engineering_path = os.path.join(base_dir, 'src', 'data', 'feature_engineering.py')
train_path = os.path.join(base_dir, 'src', 'train.py')
evaluate_path = os.path.join(base_dir, 'src', 'evaluate.py')

# Run the scripts
try:
    os.system(f"python {data_preprocessing_path}")
    os.system(f"python {feature_engineering_path}")
    os.system(f"python {train_path}")
    os.system(f"python {evaluate_path}")
except Exception as e:
    print(f"Error running command: {e}")