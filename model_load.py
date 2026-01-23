import joblib
import os
import sys
import numpy as np

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LR_MODEL_PATH = os.path.join(BASE_DIR, 'lr_model.pkl')
RF_MODEL_PATH = os.path.join(BASE_DIR, 'rf_model.pkl') # Assuming 'rd_model' refers to the random forest model

def load_lr_model(path=None):
    """
    Load the Linear Regression model and print its input parameters/features.
    Args:
        path (str, optional): Path to the .pkl file. Defaults to None (uses LR_MODEL_PATH).
    Returns:
        tuple: (model, info_str) or (None, error_str)
    """
    target_path = path if path else LR_MODEL_PATH
    print(f"\n--- Loading Linear Regression Model from {target_path} ---")
    
    if not os.path.exists(target_path):
        msg = f"Error: Model file not found at {target_path}"
        print(msg)
        return None, msg

    try:
        lr_model = joblib.load(target_path)
        
        info = []
        info.append("Model Loaded Successfully.")
        info.append(f"Source: {os.path.basename(target_path)}")
        
        # Inspect parameters
        info.append(f"Coefficients: {lr_model.coef_}")
        info.append(f"Intercept: {lr_model.intercept_}")
        
        if hasattr(lr_model, 'feature_names_in_'):
             info.append(f"Input Features: {lr_model.feature_names_in_}")
        else:
             info.append("Input Features: ['CPI_lag_1m'] (Inferred from training logic)")

        return lr_model, "\n".join(info)
    except Exception as e:
        msg = f"Failed to load LR model: {e}"
        print(msg)
        return None, msg

def load_rd_model(path=None):
    """
    Load the Random Forest model (referred to as rd_model) and print its input parameters/features.
    Args:
        path (str, optional): Path to the .pkl file. Defaults to None (uses RF_MODEL_PATH).
    Returns:
        tuple: (model, info_str) or (None, error_str)
    """
    target_path = path if path else RF_MODEL_PATH
    print(f"\n--- Loading Random Forest Model (rd_model) from {target_path} ---")
    
    if not os.path.exists(target_path):
        msg = f"Error: Model file not found at {target_path}"
        print(msg)
        return None, msg

    try:
        rf_model = joblib.load(target_path)
        info = []
        info.append("Model Loaded Successfully.")
        info.append(f"Source: {os.path.basename(target_path)}")
        
        # Inspect parameters
        # RF models don't have a single coefficient list, but we can show estimators count
        if hasattr(rf_model, 'n_estimators'):
            info.append(f"N Estimators: {rf_model.n_estimators}")
        
        if hasattr(rf_model, 'feature_names_in_'):
             info.append(f"Input Features: {rf_model.feature_names_in_}")
        else:
             info.append("Input Features: (Could not determine from model object directly)")

        return rf_model, "\n".join(info)
        
    except Exception as e:
        msg = f"Failed to load RF model: {e}"
        print(msg)
        return None, msg

if __name__ == "__main__":
    lr = load_lr_model()
    rd = load_rd_model()
