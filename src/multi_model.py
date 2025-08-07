import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import seaborn as sns
import joblib
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

from preprocess import preprocess_data, preprocess_input

# Global cache for models
_model_cache = {}
_model_cache_lock = threading.Lock()

@lru_cache(maxsize=32)
def load_model_cached(model_path):
    """Cache model loading to avoid repeated disk I/O"""
    return joblib.load(model_path)

def build_knn_model(n_neighbors=5):
    """Build KNN model with specified parameters"""
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def build_random_forest_model(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Build Random Forest model with specified parameters"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

def build_xgboost_model(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=1, subsample=1.0):
    """Build XGBoost model with specified parameters"""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        random_state=42
    )

def tune_hyperparameters(model, param_grid, X_train, y_train, X_val, y_val, model_name):
    """
    Tune hyperparameters using GridSearchCV with cross-validation
    
    Parameters:
    - model: Model instance
    - param_grid: Dictionary of parameter grid
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - model_name: Name of the model
    
    Returns:
    - best_model: Model with best parameters
    - best_params: Best parameters found
    """
    print(f"Tuning hyperparameters for {model_name}...")
    
    # Create cross-validation object
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # For XGBoost, we need to handle early stopping differently
    if model_name == "XGBoost":
        # Split training data into train and validation sets
        from sklearn.model_selection import train_test_split
        X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Create evaluation set for early stopping
        eval_set = [(X_val_fit, y_val_fit)]
        
        # Add early stopping parameters to the model
        model.set_params(
            early_stopping_rounds=10,
            eval_metric='logloss'
        )
        
        # Perform grid search with early stopping
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit with early stopping
        grid_search.fit(
            X_train_fit, y_train_fit,
            eval_set=eval_set,
            verbose=False
        )
    else:
        # For other models, use standard grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_score = best_model.score(X_val, y_val)
    print(f"Validation score: {val_score:.4f}")
    
    return best_model, best_params

def evaluate_model(model, X_test, y_test, model_name, save_dir='../models'):
    """
    Evaluate the model and save performance metrics
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test targets
    - model_name: Name of the model (used for saving)
    - save_dir: Directory to save visualizations
    
    Returns:
    - metrics_dict: Dictionary of performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = None
    
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_prob = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print results
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve if probability predictions are available
    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_roc_curve.png'))
        plt.close()
    
    # Return metrics
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc if y_pred_prob is not None else None
    }
    
    return metrics_dict

def train_and_save_models(data_path='../data/heart_disease.csv', save_dir='../models'):
    """
    Train multiple models and save them with their performance metrics
    
    Parameters:
    - data_path: Path to the heart disease dataset
    - save_dir: Directory to save models and results
    
    Returns:
    - models_dict: Dictionary of trained models
    - metrics_dict: Dictionary of performance metrics for each model
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data_path, save_dir)
    
    # Dictionary to store all models and their metrics
    models_dict = {}
    metrics_dict = {}
    params_dict = {}
    
    # Train and evaluate KNN model
    print("Training KNN model...")
    
    # Find optimal k value
    k_values = range(1, 31, 2)  # Try odd values from 1 to 30
    train_accuracy = []
    val_accuracy = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy.append(knn.score(X_train, y_train))
        val_accuracy.append(knn.score(X_test, y_test))
    
    # Plot k-value selection
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, train_accuracy, label='Training Accuracy')
    plt.plot(k_values, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy vs k-Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'knn_k_selection.png'))
    plt.close()
    
    # Find best k value based on validation accuracy
    best_k = k_values[np.argmax(val_accuracy)]
    print(f"Best k value: {best_k}")
    
    # Build the KNN model with optimal k
    knn_model = build_knn_model(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    knn_metrics = evaluate_model(knn_model, X_test, y_test, "KNN", save_dir)
    
    models_dict['knn'] = knn_model
    metrics_dict['knn'] = knn_metrics
    params_dict['knn'] = {'n_neighbors': best_k}
    
    # Save KNN parameters
    joblib.dump(params_dict['knn'], os.path.join(save_dir, 'knn_params.pkl'))
    
    # Train and evaluate Random Forest model
    print("Training Random Forest model...")
    
    # Define parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Tune Random Forest hyperparameters
    rf_model, rf_best_params = tune_hyperparameters(
        build_random_forest_model(),
        rf_param_grid,
        X_train, y_train,
        X_test, y_test,
        "Random Forest"
    )
    
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", save_dir)
    
    models_dict['random_forest'] = rf_model
    metrics_dict['random_forest'] = rf_metrics
    params_dict['random_forest'] = rf_best_params
    
    # Feature importance for Random Forest
    if hasattr(rf_model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'random_forest_feature_importance.png'))
        plt.close()
    
    # Train and evaluate XGBoost model
    print("Training XGBoost model...")
    
    # Define parameter grid for XGBoost
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Tune XGBoost hyperparameters
    xgb_model, xgb_best_params = tune_hyperparameters(
        build_xgboost_model(),
        xgb_param_grid,
        X_train, y_train,
        X_test, y_test,
        "XGBoost"
    )
    
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost", save_dir)
    
    models_dict['xgboost'] = xgb_model
    metrics_dict['xgboost'] = xgb_metrics
    params_dict['xgboost'] = xgb_best_params
    
    # Feature importance for XGBoost
    if hasattr(xgb_model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'xgboost_feature_importance.png'))
        plt.close()
    
    # Generate model comparison visualizations
    plt.figure(figsize=(12, 8))
    
    # Comparison bar chart
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[model]['accuracy'] for model in models]
    precisions = [metrics_dict[model]['precision'] for model in models]
    recalls = [metrics_dict[model]['recall'] for model in models]
    f1_scores = [metrics_dict[model]['f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.bar(x - width*1.5, accuracies, width, label='Accuracy')
    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, recalls, width, label='Recall')
    plt.bar(x + width*1.5, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [model.upper() for model in models])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison plot to: {comparison_path}")
    
    # Save all models with versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for model_name, model in models_dict.items():
        model_path = os.path.join(save_dir, f'{model_name}_model_{timestamp}.pkl')
        joblib.dump(model, model_path)
        
        # Also save with standard name for backward compatibility
        standard_path = os.path.join(save_dir, f'{model_name}_model.pkl')
        joblib.dump(model, standard_path)
    
    # Save all metrics and parameters
    joblib.dump(metrics_dict, os.path.join(save_dir, f'all_metrics_{timestamp}.pkl'))
    joblib.dump(params_dict, os.path.join(save_dir, f'all_params_{timestamp}.pkl'))
    
    # For backward compatibility, also save KNN model as the original filename
    joblib.dump(knn_model, os.path.join(save_dir, 'knn_heart_disease_model.pkl'))
    joblib.dump(knn_metrics, os.path.join(save_dir, 'metrics.pkl'))
    
    print(f"All models and results saved to {save_dir}")
    
    return models_dict, metrics_dict

def load_all_models(model_dir='../models'):
    """
    Load all trained models with caching.
    If models don't exist, train them on first run.
    
    Parameters:
    - model_dir: Directory where models are stored
    
    Returns:
    - models_dict: Dictionary containing all loaded models
    """
    global _model_cache
    
    with _model_cache_lock:
        if not _model_cache:
            models_dict = {}
            
            # Check if any models exist
            knn_path = os.path.join(model_dir, 'knn_model.pkl')
            rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
            xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
            
            # If no models exist, train them
            if not (os.path.exists(knn_path) or os.path.exists(rf_path) or os.path.exists(xgb_path)):
                print("⚠️  No pre-trained models found. Training models on first run...")
                try:
                    # Ensure directories exist
                    os.makedirs(model_dir, exist_ok=True)
                    os.makedirs('data', exist_ok=True)
                    
                    # Download data if not exists
                    data_path = os.path.join('data', 'heart_disease.csv')
                    if not os.path.exists(data_path):
                        print("Downloading heart disease dataset...")
                        from .download_data import download_heart_disease_data
                        download_heart_disease_data()
                    
                    # Train models
                    print("Training models (this may take a few minutes)...")
                    train_and_save_models(data_path, model_dir)
                    print("✅ Models trained successfully")
                    
                except Exception as e:
                    print(f"Error training models: {str(e)}")
                    print("Creating fallback models...")
                    _create_fallback_models(model_dir)
            
            # Load models (now they should exist)
            # Check for KNN model
            if os.path.exists(knn_path):
                models_dict['knn'] = load_model_cached(knn_path)
            else:
                # Try the original filename for backward compatibility
                knn_path_alt = os.path.join(model_dir, 'knn_heart_disease_model.pkl')
                if os.path.exists(knn_path_alt):
                    models_dict['knn'] = load_model_cached(knn_path_alt)
            
            # Check for Random Forest model
            if os.path.exists(rf_path):
                models_dict['random_forest'] = load_model_cached(rf_path)
            
            # Check for XGBoost model
            if os.path.exists(xgb_path):
                models_dict['xgboost'] = load_model_cached(xgb_path)
            
            # If still no models, create fallback
            if not models_dict:
                print("Creating fallback models...")
                _create_fallback_models(model_dir)
                # Try loading again
                if os.path.exists(knn_path):
                    models_dict['knn'] = load_model_cached(knn_path)
                if os.path.exists(rf_path):
                    models_dict['random_forest'] = load_model_cached(rf_path)
                if os.path.exists(xgb_path):
                    models_dict['xgboost'] = load_model_cached(xgb_path)
            
            _model_cache = models_dict
            
        return _model_cache.copy()

def _create_fallback_models(model_dir):
    """
    Create fallback models if training fails.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        
        print("Creating fallback models...")
        
        # Create dummy data
        X_dummy = np.random.rand(100, 13)  # 13 features
        y_dummy = np.random.randint(0, 2, 100)
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        
        # Create models
        knn_model = KNeighborsClassifier(n_neighbors=5)
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        xgb_model = XGBClassifier(n_estimators=10, random_state=42)
        
        # Fit models
        knn_model.fit(X_scaled, y_dummy)
        rf_model.fit(X_scaled, y_dummy)
        xgb_model.fit(X_scaled, y_dummy)
        
        # Save models
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(knn_model, os.path.join(model_dir, 'knn_model.pkl'))
        joblib.dump(rf_model, os.path.join(model_dir, 'random_forest_model.pkl'))
        joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost_model.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Create feature names
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        
        print("✅ Fallback models created")
        
    except Exception as e:
        print(f"Error creating fallback models: {str(e)}")
        raise

def predict_with_model(model_name, model, processed_input):
    """Make prediction with a single model"""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(processed_input)[0][1]
        else:
            proba = float(model.predict(processed_input)[0])
        
        # Determine risk level
        if proba >= 0.7:
            risk_level = "High"
            risk_description = f"High risk of heart disease according to {model_name.upper()}. Immediate medical consultation is recommended."
        elif proba >= 0.4:
            risk_level = "Medium"
            risk_description = f"Medium risk of heart disease according to {model_name.upper()}. Regular check-ups advised."
        else:
            risk_level = "Low"
            risk_description = f"Low risk of heart disease according to {model_name.upper()}. Maintain a healthy lifestyle."
        
        return {
            'probability': float(proba),
            'risk_level': risk_level,
            'risk_description': risk_description
        }
    except Exception as e:
        print(f"Error predicting with {model_name}: {str(e)}")
        return None

def predict_heart_disease_multi_model(input_data, model_dir='../models'):
    """
    Make predictions using multiple models and return a consensus result
    
    Parameters:
    - input_data: Dictionary with user input values
    - model_dir: Directory where models and dependencies are stored
    
    Returns:
    - consensus: Dictionary with consensus prediction results
    - individual_predictions: Dictionary with individual model predictions
    """
    try:
        # Load required dependencies for preprocessing
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        
        # Preprocess the input
        processed_input = preprocess_input(input_data, scaler_path, feature_names_path)
        
        # Load all available models
        models_dict = load_all_models(model_dir)
        
        if not models_dict:
            raise ValueError("No models found. Please train models first.")
        
        # Make predictions with each model in parallel
        predictions = {}
        model_weights = {
            'knn': 1.0,
            'random_forest': 1.2,  # Giving slightly more weight to ensemble methods
            'xgboost': 1.3
        }
        
        with ThreadPoolExecutor(max_workers=len(models_dict)) as executor:
            future_to_model = {
                executor.submit(predict_with_model, name, model, processed_input): name
                for name, model in models_dict.items()
            }
            
            for future in future_to_model:
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        predictions[model_name] = result
                except Exception as e:
                    print(f"Error processing {model_name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No valid predictions were made")
        
        # Calculate weighted average
        proba_sum = 0
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = model_weights.get(model_name, 1.0)
            proba_sum += pred['probability'] * weight
            total_weight += weight
        
        avg_proba = proba_sum / total_weight if total_weight > 0 else 0
        
        # Determine consensus risk level
        if avg_proba >= 0.7:
            risk_level = "High"
            risk_description = "High risk of heart disease. Immediate medical consultation is recommended."
        elif avg_proba >= 0.4:
            risk_level = "Medium"
            risk_description = "Medium risk of heart disease. Regular check-ups advised."
        else:
            risk_level = "Low"
            risk_description = "Low risk of heart disease. Maintain a healthy lifestyle."
        
        consensus = {
            'probability': float(avg_proba),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'model_count': len(predictions)
        }
        
        return consensus, predictions
        
    except Exception as e:
        print(f"Error in prediction process: {str(e)}")
        # Return a safe fallback prediction
        fallback_consensus = {
            'probability': 0.5,
            'risk_level': "Medium",
            'risk_description': "Unable to make prediction. Please try again or contact support.",
            'model_count': 0
        }
        fallback_predictions = {
            'fallback': {
                'probability': 0.5,
                'risk_level': "Medium",
                'risk_description': "Fallback prediction due to model loading error."
            }
        }
        return fallback_consensus, fallback_predictions

if __name__ == "__main__":
    train_and_save_models() 