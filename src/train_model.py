import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import joblib
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

from preprocess import preprocess_data

def find_optimal_k(X_train, y_train, k_range=range(1, 31, 2), n_jobs=-1):
    """
    Find optimal k value for KNN using parallel processing
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - k_range: Range of k values to try
    - n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
    - optimal_k: Best k value
    - cv_scores: Cross-validation scores
    """
    try:
        cv_scores = []
        
        def evaluate_k(k):
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=5, n_jobs=1)
            return k, np.mean(scores)
        
        # Use parallel processing to evaluate k values
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            results = list(executor.map(evaluate_k, k_range))
        
        # Sort results by k value
        results.sort(key=lambda x: x[0])
        k_values, cv_scores = zip(*results)
        
        # Find best k
        optimal_k = k_values[np.argmax(cv_scores)]
        
        return optimal_k, cv_scores
        
    except Exception as e:
        print(f"Error finding optimal k: {str(e)}")
        raise

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Plot and save confusion matrix
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - save_dir: Directory to save the plot
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")
        raise

def train_quick_assessment_model(data_path='data/heart_disease.csv', save_dir='models', n_jobs=-1):
    """
    Train a quick assessment model using only essential features.
    
    Parameters:
    - data_path: Path to the dataset
    - save_dir: Directory to save model and dependencies
    - n_jobs: Number of parallel jobs for training
    
    Returns:
    - metrics: Dictionary containing model performance metrics
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Define quick assessment features (only the essential ones)
        quick_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'restecg']
        
        # Select only the quick assessment features
        X = df[quick_features]
        y = df['target']
        
        # Split the data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Find optimal k using parallel processing
        optimal_k, cv_scores = find_optimal_k(X_train_scaled, y_train, n_jobs=n_jobs)
        
        # Train model with optimal k
        knn = KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=n_jobs)
        knn.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'optimal_k': optimal_k
        }
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, save_dir)
        
        # Save model and dependencies
        model_path = os.path.join(save_dir, 'quick_assessment_model.pkl')
        scaler_path = os.path.join(save_dir, 'quick_assessment_scaler.pkl')
        feature_names_path = os.path.join(save_dir, 'quick_assessment_features.pkl')
        metrics_path = os.path.join(save_dir, 'quick_assessment_metrics.pkl')
        
        joblib.dump(knn, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(quick_features, feature_names_path)
        joblib.dump(metrics, metrics_path)
        
        print(f"Quick Assessment Model trained and saved successfully!")
        print(f"Features used: {quick_features}")
        print(f"Optimal k value: {optimal_k}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error in quick assessment training process: {str(e)}")
        raise

def train_and_save_model(data_path='data/heart_disease.csv', save_dir='models', n_jobs=-1):
    """
    Train KNN model and save it along with its dependencies.
    
    Parameters:
    - data_path: Path to the dataset
    - save_dir: Directory to save model and dependencies
    - n_jobs: Number of parallel jobs for training
    
    Returns:
    - metrics: Dictionary containing model performance metrics
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(data_path, save_dir)
        
        # Find optimal k using parallel processing
        optimal_k, cv_scores = find_optimal_k(X_train, y_train, n_jobs=n_jobs)
        
        # Train model with optimal k
        knn = KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=n_jobs)
        knn.fit(X_train, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'optimal_k': optimal_k
        }
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, save_dir)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
        plt.xlabel('Cross-validation fold')
        plt.ylabel('Accuracy')
        plt.title('Training History')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()
        
        # Save model and metrics
        model_path = os.path.join(save_dir, 'knn_heart_disease_model.pkl')
        metrics_path = os.path.join(save_dir, 'metrics.pkl')
        
        joblib.dump(knn, model_path)
        joblib.dump(metrics, metrics_path)
        
        print(f"Model trained and saved successfully!")
        print(f"Optimal k value: {optimal_k}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    # Train both models
    print("Training full model...")
    train_and_save_model()
    print("\n" + "="*50 + "\n")
    print("Training quick assessment model...")
    train_quick_assessment_model() 