import os
import subprocess
import mlflow

# def get_script_path(script_name):
#     """
#     Construct the path to the script relative to the current file's directory.
#     """
#     return os.path.join(os.path.dirname(__file__), script_name)

def run_command(command):
    """
    Run a shell command and capture its output and errors.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)  # Print the standard output
        if result.stderr:
            print(f"Error: {result.stderr}")  # Print the error output if there's any
    except Exception as e:
        print(f"Error running command: {e}")

def preprocess_data():
    """
    Run the data fetching, preprocessing, and feature engineering scripts.
    """
    fetch_data_path = "fetch_data.py"
    data_preprocessing_script = "data_preprocessing.py"
    feature_engineering_script = "feature_engineering.py"

    print("Preprocessing data...")
    run_command(f"python {fetch_data_path}")
    run_command(f"python {data_preprocessing_script}")
    run_command(f"python {feature_engineering_script}")

def train_models():
    """
    Run the training script to train all models.
    """
    print("Training models...")
    train_script = 'train.py'
    
    # Check if file exists before running command
    if os.path.exists(train_script):
        run_command(f"python {train_script}")
    else:
        print(f"File not found: {train_script}")

def evaluate_models():
    """
    Run the evaluation script to evaluate all trained models.
    """
    print("Evaluating models...")
    evaluate_script = 'evaluate.py'
    
    # Check if file exists before running command
    if os.path.exists(evaluate_script):
        run_command(f"python {evaluate_script}")
    else:
        print(f"File not found: {evaluate_script}")

def log_pipeline_to_mlflow():
    """
    Log the pipeline stages to MLflow for tracking.
    """
    mlflow.set_experiment("Stock Forecasting Pipeline")  # Set experiment before starting run
    
    with mlflow.start_run(run_name="Stock Forecasting Pipeline") as run:
        # Stage 1: Data Preprocessing
        mlflow.log_param("Stage", "Data Preprocessing")
        try:
            preprocess_data()
        except Exception as e:
            print(f"Error running data preprocessing: {e}")

        # Stage 2: Model Training
        with mlflow.start_run(nested=True):
            mlflow.log_param("Stage", "Model Training")
            try:
                train_models()
            except Exception as e:
                print(f"Error running model training: {e}")

        # Stage 3: Model Evaluation
        with mlflow.start_run(nested=True):
            mlflow.log_param("Stage", "Model Evaluation")
            try:
                evaluate_models()
            except Exception as e:
                print(f"Error running model evaluation: {e}")

if __name__ == "__main__":
    log_pipeline_to_mlflow()
