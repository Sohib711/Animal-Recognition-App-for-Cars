import platform
import sys
import os
import tensorflow as tf
import keras
from typing import Dict, Any
from numpy import mean, std, ndarray
from pandas import DataFrame, read_csv

def is_gpu_active() -> None:
    """
    Check if a GPU is available for TensorFlow and display system information.

    Prints:
        - Python platform information.
        - TensorFlow and Keras versions.
        - Python version.
        - GPU availability status.
    """
    
    print(f"Python Platform: {platform.platform()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print(f"Python Version: {sys.version}\n")
    
    try:
        gpu = len(tf.config.list_physical_devices('GPU')) > 0
        print("GPU is", "available" if gpu else "NOT AVAILABLE")
    except Exception as e:
        print(f"Error checking GPU availability: {e}")



def calculate_mean_std(data: ndarray) -> tuple:
    """
    Calculate the mean and standard deviation of a given dataset.

    Args:
        data (np.ndarray): The input data array.

    Returns:
        tuple: A tuple containing:
               - mean (float): The mean of the data.
               - std (float): The standard deviation of the data.
    """
    
    mean_value = mean(data)
    std_value = std(data)
    return mean_value, std_value


def initialize_report(csv_file_path: str) -> DataFrame:
    """
    Initializes a report CSV file if it does not exist.

    Args:
        csv_file_path (str): The path to the CSV file for the report.

    Returns:
        DataFrame: A DataFrame containing the report structure.
    """
    os.makedirs("Results", exist_ok=True)

    if not os.path.exists(csv_file_path):
        report: Dict[str, Any] = {
            'Model': [],
            'accuracy Mean': [],
            'accuracy Std': [],
            'loss Mean': [],
            'loss Std': [],
            'precision Mean': [],
            'precision Std': [],
            'val_accuracy Mean': [],
            'val_accuracy Std': [],
            'val_loss Mean': [],
            'val_loss Std': [],
            'val_precision Mean': [],
            'val_precision Std': [],
            'Description': []
        }
        df_report = DataFrame(report)
        df_report.to_csv(csv_file_path, index=False)
    else:
        df_report = read_csv(csv_file_path)
    
    return df_report