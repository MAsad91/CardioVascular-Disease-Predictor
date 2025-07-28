import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from functools import lru_cache
import threading

# Global cache for scaler and feature names
_scaler_cache = None
_feature_names_cache = None
_cache_lock = threading.Lock()

def validate_input_data(input_data, feature_names):
    """
    Validate input data against expected features
    
    Parameters:
    - input_data: Dictionary with input values
    - feature_names: List of expected feature names
    
    Returns:
    - bool: True if input is valid, False otherwise
    """
    # Check if all required features are present
    missing_features = set(feature_names) - set(input_data.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for invalid values
    for feature, value in input_data.items():
        if not isinstance(value, (int, float)):
            try:
                input_data[feature] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for feature {feature}: {value}")
    
    return True

@lru_cache(maxsize=32)
def load_scaler_cached(scaler_path):
    """Cache scaler loading to avoid repeated disk I/O"""
    return joblib.load(scaler_path)

@lru_cache(maxsize=32)
def load_feature_names_cached(feature_names_path):
    """Cache feature names loading to avoid repeated disk I/O"""
    return joblib.load(feature_names_path)

def preprocess_data(data_path='data/heart_disease.csv', save_dir='models'):
    """
    Preprocess the heart disease dataset.
    
    Parameters:
    - data_path: Path to the dataset
    - save_dir: Directory to save preprocessing objects
    
    Returns:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - feature_names: List of feature names
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Validate data
        if df.isnull().any().any():
            raise ValueError("Dataset contains missing values")
        
        # Extract features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Save feature names
        feature_names = list(X.columns)
        joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
        
    except Exception as e:
        print(f"Error in preprocessing data: {str(e)}")
        raise

def prepare_sequence_data(X, y, sequence_length=5):
    """
    Prepare sequence data for RNN/LSTM models. This function is used
    to create synthetic sequence data since heart disease data isn't 
    naturally sequential. For real applications, you would use actual
    sequential patient data.
    
    Parameters:
    - X: Features array
    - y: Target array
    - sequence_length: Length of sequences to create
    
    Returns:
    - X_seq: Sequence data (3D array)
    - y_seq: Corresponding targets
    """
    X_seq = []
    y_seq = []
    
    # For demonstration, create synthetic sequences by grouping samples
    # In a real application, you would use actual time-series patient data
    for i in range(0, len(X) - sequence_length + 1, 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y.iloc[i+sequence_length-1] if isinstance(y, pd.Series) else y[i+sequence_length-1])
    
    return np.array(X_seq), np.array(y_seq)

def preprocess_input(input_data, scaler_path, feature_names_path):
    """
    Preprocess input data for prediction with caching and validation
    
    Parameters:
    - input_data: Dictionary with input values
    - scaler_path: Path to saved scaler
    - feature_names_path: Path to saved feature names
    
    Returns:
    - processed_input: Preprocessed input ready for prediction
    """
    global _scaler_cache, _feature_names_cache
    
    try:
        with _cache_lock:
            # Load scaler and feature names from cache or disk
            if _scaler_cache is None:
                _scaler_cache = load_scaler_cached(scaler_path)
            if _feature_names_cache is None:
                _feature_names_cache = load_feature_names_cached(feature_names_path)
        
        # Validate input data
        validate_input_data(input_data, _feature_names_cache)
        
        # Convert input to DataFrame with correct feature order
        input_df = pd.DataFrame([input_data], columns=_feature_names_cache)
        
        # Scale input
        scaled_input = _scaler_cache.transform(input_df)
        
        return scaled_input
        
    except Exception as e:
        print(f"Error preprocessing input: {str(e)}")
        raise 