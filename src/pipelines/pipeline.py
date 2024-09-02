import os
import subprocess
import mlflow

def run_command(command):
    """
    Utility function to run shell commands.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\n{stderr.decode('utf-8')}")
    else:
        print(stdout.decode('utf-8'))

def get_script_path(script_name):
    """
    Construct the path to the script relative to the current file's directory.
    """
    return os.path.join(os.path.dirname(__file__), '..', '..', script_name)

def preprocess_data():
    """
    Run the feature engineering script to preprocess data.
    """
    print("Preprocessing data...")
    fetch_data_path = get_script_path('data/fetch_data.py')
    data_preprocessing_script = get_script_path('data/data_preprocessing.py')
    feature_engineering_script = get_script_path('data/feature_engineering.py')
    run_command(f"python {fetch_data_path}")
    run_command(f"python {data_preprocessing_script}")
    run_command(f"python {feature_engineering_script}")

def train_models():
    """
    Run the training script to train all models.
    """
    print("Training models...")
    train_script = get_script_path('train.py')
    run_command(f"python {train_script}")

def evaluate_models():
    """
    Run the evaluation script to evaluate all trained models.
    """
    print("Evaluating models...")
    evaluate_script = get_script_path('evaluate.py')
    run_command(f"python {evaluate_script}")

def log_pipeline_to_mlflow():
    with mlflow.start_run(run_name="Stock Forecasting Pipeline") as run:
        # Stage 1: Data Preprocessing
        mlflow.log_param("Stage", "Data Preprocessing")
        try:
            preprocess_data()
        except Exception as e:
            print(f"Error running data preprocessing: {e}")

        # Stage 2: Feature Engineering
        with mlflow.start_run(nested=True) as nested_run:
            mlflow.log_param("Stage", "Feature Engineering")
            try:
                preprocess_data()
            except Exception as e:
                print(f"Error running feature engineering: {e}")

        # Stage 3: Model Training
        with mlflow.start_run(nested=True) as nested_run:
            mlflow.log_param("Stage", "Model Training")
            try:
                train_models()
            except Exception as e:
                print(f"Error running model training: {e}")

        # Stage 4: Model Evaluation
        with mlflow.start_run(nested=True) as nested_run:
            mlflow.log_param("Stage", "Model Evaluation")
            try:
                evaluate_models()
            except Exception as e:
                print(f"Error running model evaluation: {e}")

if __name__ == "__main__":
    print("Running the Stock Forecasting Pipeline...")
    log_pipeline_to_mlflow()
    print("Pipeline execution completed.")
