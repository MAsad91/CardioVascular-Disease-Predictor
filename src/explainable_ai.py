import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import base64
from io import BytesIO
import uuid

def get_feature_importance(model_dir='models', model_type=None):
    """
    Calculate feature importance using permutation importance.
    
    Parameters:
    - model_dir: Directory where the model and dependencies are stored
    - model_type: Type of model to use for feature importance ('knn', 'random_forest', 'xgboost')
    
    Returns:
    - importance_df: DataFrame with feature importances
    """
    try:
        print(f"DEBUG: get_feature_importance called with model_dir={model_dir}, model_type={model_type}")
        
        # Determine which model to use
        if model_type is None:
            # Default to Random Forest for better feature importance
            model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        else:
            # Use specific model type
            if model_type == 'knn':
                model_path = os.path.join(model_dir, 'knn_heart_disease_model.pkl')
            elif model_type == 'random_forest':
                model_path = os.path.join(model_dir, 'random_forest_model.pkl')
            elif model_type == 'xgboost':
                model_path = os.path.join(model_dir, 'xgboost_model.pkl')
            else:
                # Default to Random Forest if model type is not recognized
                model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        
        print(f"DEBUG: Using model path: {model_path}")
        print(f"DEBUG: Model file exists: {os.path.exists(model_path)}")
        
        # Load the model
        model = joblib.load(model_path)
        print(f"DEBUG: Model loaded successfully, type: {type(model)}")
        
        # Load the feature names
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        print(f"DEBUG: Feature names path: {feature_names_path}")
        print(f"DEBUG: Feature names file exists: {os.path.exists(feature_names_path)}")
        
        feature_names = joblib.load(feature_names_path)
        print(f"DEBUG: Feature names loaded: {feature_names}")
        
        # Load training data
        data_path = os.path.join('data', 'heart_disease.csv')
        print(f"DEBUG: Data path: {data_path}")
        print(f"DEBUG: Data file exists: {os.path.exists(data_path)}")
        
        df = pd.read_csv(data_path)
        print(f"DEBUG: Data loaded, shape: {df.shape}")
        
        # Prepare data
        X_train = df.drop('target', axis=1).values
        y_train = df['target'].values
        print(f"DEBUG: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # If the model has built-in feature importance (RF or XGBoost), use it
        if hasattr(model, 'feature_importances_'):
            print(f"DEBUG: Using built-in feature importance")
            importances = model.feature_importances_
            std = np.zeros_like(importances)  # No std for built-in importance
        else:
            print(f"DEBUG: Using permutation importance")
            # For KNN or if built-in importance isn't available, use permutation importance
            result = permutation_importance(
                model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
            std = result.importances_std
        
        print(f"DEBUG: Importances calculated, length: {len(importances)}")
        print(f"DEBUG: Importances: {importances}")
        
        # Create a DataFrame with the feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'StdDev': std
        })
        
        print(f"DEBUG: Importance DataFrame created, shape: {importance_df.shape}")
        print(f"DEBUG: Importance DataFrame head:\n{importance_df.head()}")
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Normalize importance scores to sum to 1
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
        
        print(f"DEBUG: Final importance DataFrame:\n{importance_df}")
        
        return importance_df
        
    except Exception as e:
        print(f"Error in get_feature_importance: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a default importance DataFrame if there's an error
        default_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        default_df = pd.DataFrame({
            'Feature': default_features,
            'Importance': [1/len(default_features)] * len(default_features),
            'StdDev': [0] * len(default_features)
        })
        print(f"DEBUG: Returning default importance DataFrame:\n{default_df}")
        return default_df

def get_neighbors_data(input_data, model_dir='models'):
    """
    Get data about the K nearest neighbors used for the prediction.
    
    Parameters:
    - input_data: Dictionary with user input values
    - model_dir: Directory where the model and dependencies are stored
    
    Returns:
    - neighbors_df: DataFrame with information about the neighbors
    """
    # Load the model and other dependencies
    model_path = os.path.join(model_dir, 'knn_heart_disease_model.pkl')
    # Check if model exists, if not, return empty DataFrame
    if not os.path.exists(model_path):
        print(f"Warning: KNN model not found at {model_path}. Returning empty neighbors data.")
        return pd.DataFrame() # Return an empty DataFrame to avoid errors
    
    model = joblib.load(model_path)
    
    # Load the feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    # Check if feature names exist, if not, return empty DataFrame
    if not os.path.exists(feature_names_path):
        print(f"Warning: Feature names not found at {feature_names_path}. Returning empty neighbors data.")
        return pd.DataFrame()

    feature_names = joblib.load(feature_names_path)
    
    # Load the scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    # Check if scaler exists, if not, return empty DataFrame
    if not os.path.exists(scaler_path):
        print(f"Warning: Scaler not found at {scaler_path}. Returning empty neighbors data.")
        return pd.DataFrame()
    scaler = joblib.load(scaler_path)
    
    # Load training data
    data_path = os.path.join('data', 'heart_disease.csv')
    # Check if data exists, if not, return empty DataFrame
    if not os.path.exists(data_path):
        print(f"Warning: Training data not found at {data_path}. Returning empty neighbors data.")
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    
    # Prepare data
    X_train = df.drop('target', axis=1).values
    y_train = df['target'].values
    
    # Ensure input_data has all features and is in correct order
    input_df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0 # Add missing columns with a default value
    input_df = input_df[feature_names]  # Ensure correct order
    X = scaler.transform(input_df)
    
    # Get the indices of the k nearest neighbors
    try:
        _, indices = model.kneighbors(X)
        indices = indices[0]
    except Exception as e:
        print(f"Error getting k-neighbors: {str(e)}. Returning empty neighbors data.")
        return pd.DataFrame() # Return empty DataFrame on error
    
    # Get the data for the neighbors
    neighbors_X = X_train[indices]
    neighbors_y = y_train[indices]
    
    # Calculate similarity score for each neighbor (1 - normalized distance)
    distances = []
    for idx in indices:
        # Euclidean distance
        dist = np.linalg.norm(X - X_train[idx])
        distances.append(dist)
    
    # Normalize distances to 0-1 range
    max_dist = max(distances) if max(distances) > 0 else 1
    similarity_scores = [1 - (d / max_dist) for d in distances]
    
    # Create a DataFrame for neighbors
    neighbors_data = []
    for i, idx in enumerate(indices):
        neighbor_data = {
            'Neighbor': i + 1,
            'Prediction': 'Positive' if neighbors_y[i] == 1 else 'Negative',
            'Probability': model.predict_proba(X_train[idx].reshape(1, -1))[0][1] if hasattr(model, 'predict_proba') else 0.0, # Add probability for neighbor
            'Similarity': similarity_scores[i] # Keep as float for sorting
        }
        
        # Add feature values
        for j, feature in enumerate(feature_names):
            neighbor_data[feature] = neighbors_X[i][j]
        
        neighbors_data.append(neighbor_data)
    
    neighbors_df = pd.DataFrame(neighbors_data)
    
    return neighbors_df

def create_feature_importance_plot(importance_df, top_n=10):
    """
    Create a plot showing the importance of features.
    
    Parameters:
    - importance_df: DataFrame with feature importances
    - top_n: Number of top features to show
    
    Returns:
    - fig: Matplotlib figure
    """
    try:
        print(f"DEBUG: create_feature_importance_plot called with importance_df type: {type(importance_df)}")
        
        # Clear any existing plots
        plt.clf()
        
        # Ensure importance_df is a DataFrame
        if not isinstance(importance_df, pd.DataFrame):
            print(f"Warning: importance_df is not a DataFrame, got {type(importance_df)}. Cannot create feature importance plot.")
            return None
        
        print(f"DEBUG: importance_df shape: {importance_df.shape}")
        print(f"DEBUG: importance_df columns: {importance_df.columns.tolist()}")
        print(f"DEBUG: importance_df head:\n{importance_df.head()}")
        
        # Ensure required columns exist
        required_columns = ['Feature', 'Importance']
        if not all(col in importance_df.columns for col in required_columns):
            print(f"Warning: importance_df missing required columns. Found: {importance_df.columns}. Cannot create feature importance plot.")
            return None
        
        # Get the top N features
        top_features = importance_df.head(top_n).copy()
        
        if top_features.empty:
            print("Warning: No features found in importance_df. Cannot create feature importance plot.")
            return None
        
        print(f"DEBUG: top_features shape: {top_features.shape}")
        print(f"DEBUG: top_features:\n{top_features}")
        
        # Sort by importance in descending order
        top_features = top_features.sort_values('Importance', ascending=True)
        
        # Create the plot using matplotlib directly instead of seaborn for better compatibility
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create horizontal bar plot
        y_pos = range(len(top_features))
        bars = ax.barh(y_pos, top_features['Importance'], color='skyblue', alpha=0.8, height=0.6)
        
        # Add value labels on the bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
            ax.text(importance + 0.01, i, f'{importance:.3f}', va='center', fontweight='bold', fontsize=9)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # Add grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent stretching
        plt.tight_layout(pad=1.5)
        
        print(f"DEBUG: Feature importance plot created successfully")
        return fig
        
    except Exception as e:
        print(f"Error in create_feature_importance_plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_feature_comparison_plot(neighbors_df, input_data, top_n=5):
    """
    Create a plot comparing the patient's values with the neighbors.
    
    Parameters:
    - neighbors_df: DataFrame with neighbor data
    - input_data: Patient's data
    - top_n: Number of top features to show
    
    Returns:
    - fig: Matplotlib figure
    """
    # Clear any existing plots
    plt.clf()

    if neighbors_df.empty:
        print("Warning: neighbors_df is empty. Cannot create feature comparison plot.")
        return None
        
    # Select top features to visualize
    feature_names = [col for col in neighbors_df.columns if col not in ['Neighbor', 'Prediction', 'Similarity', 'Probability']] # Exclude 'Probability'
    top_features = feature_names[:top_n]  # Just use first 5 for visualization
    
    # Create the plot
    fig, axes = plt.subplots(len(top_features), 1, figsize=(10, 3*len(top_features)))
    
    # If only one feature, axes is not an array
    if len(top_features) == 1:
        axes = [axes]
    
    # For each feature
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        # Plot neighbors' values as box plot
        sns.boxplot(x=neighbors_df[feature], ax=ax, color='skyblue', boxprops=dict(alpha=0.7), medianprops=dict(color='orange'))
        
        # Plot patient's value as a red dot
        patient_value = input_data.get(feature, None)
        if patient_value is not None:
            ax.plot(patient_value, 0, 'o', color='red', markersize=8, label='Your Value')
        
        # Customize the plot
        ax.set_title(f'Comparison for {feature.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('', fontsize=10) # No y-label for box plots
        ax.legend()
        
        # Remove y-axis ticks and labels for cleaner box plots
        ax.set(yticks=[], yticklabels=[])
        
    plt.tight_layout()
    return fig

def create_key_risk_factors_chart(input_data, feature_importance):
    """
    Create a horizontal bar chart showing key risk factors with scores from 0-100.
    This matches the "Your Key Risk Factors" chart format.
    
    Parameters:
    - input_data: Dictionary with user input values
    - feature_importance: DataFrame with feature importances
    
    Returns:
    - fig: Matplotlib figure
    """
    try:
        # Clear all matplotlib state
        plt.close('all')
        plt.clf()
        plt.cla()
        
        # Ensure feature_importance is a DataFrame
        if not isinstance(feature_importance, pd.DataFrame):
            print(f"Warning: feature_importance is not a DataFrame, got {type(feature_importance)}. Cannot create key risk factors chart.")
            return None
        
        # Define feature mappings and ranges for risk scoring
        feature_ranges = {
            'age': {'min': 20, 'max': 100, 'risk_threshold': 60},
            'sex': {'min': 0, 'max': 1, 'risk_threshold': 1},  # Male is higher risk
            'cp': {'min': 0, 'max': 3, 'risk_threshold': 0},  # Typical angina is highest risk
            'trestbps': {'min': 90, 'max': 200, 'risk_threshold': 140},
            'chol': {'min': 100, 'max': 600, 'risk_threshold': 240},
            'fbs': {'min': 0, 'max': 1, 'risk_threshold': 1},  # Yes is higher risk
            'restecg': {'min': 0, 'max': 2, 'risk_threshold': 1},
            'thalach': {'min': 60, 'max': 220, 'risk_threshold': 120, 'inverse': True},  # Lower is worse
            'exang': {'min': 0, 'max': 1, 'risk_threshold': 1},  # Yes is higher risk
            'oldpeak': {'min': 0, 'max': 6.2, 'risk_threshold': 1.0},
            'slope': {'min': 0, 'max': 2, 'risk_threshold': 2},  # Downsloping is highest risk
            'ca': {'min': 0, 'max': 3, 'risk_threshold': 1},
            'thal': {'min': 1, 'max': 3, 'risk_threshold': 3}  # Reversible defect is highest risk
        }
        
        # Feature name mappings for display
        feature_names = {
            'age': 'Age',
            'sex': 'Sex',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting BP',
            'chol': 'Cholesterol',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate',
            'exang': 'Exercise Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels',
            'thal': 'Thalassemia'
        }
        
        # Calculate risk scores for each feature
        risk_scores = []
        feature_labels = []
        
        # Sort features by importance and take top 6
        if 'Feature' in feature_importance.columns and 'Importance' in feature_importance.columns:
            top_features = feature_importance.nlargest(6, 'Importance')['Feature'].tolist()
        else:
            # Fallback to most important features if columns are missing
            top_features = ['thal', 'ca', 'oldpeak', 'trestbps', 'chol', 'thalach']
        
        for feature in top_features:
            if feature in input_data and feature in feature_ranges:
                value = input_data[feature]
                range_info = feature_ranges[feature]
                
                # Calculate risk score (0-100)
                if range_info.get('inverse', False):
                    # For inverse features (like thalach), lower values = higher risk
                    normalized = (range_info['max'] - value) / (range_info['max'] - range_info['min'])
                else:
                    # For normal features, higher values = higher risk
                    normalized = (value - range_info['min']) / (range_info['max'] - range_info['min'])
                
                # Convert to 0-100 scale and clamp
                risk_score = max(0, min(100, normalized * 100))
                
                # Special adjustments for certain features
                if feature == 'age' and value < 40:
                    risk_score = max(0, risk_score - 20)  # Lower risk for younger people
                elif feature == 'trestbps' and value < 120:
                    risk_score = max(0, risk_score - 30)  # Lower risk for normal BP
                elif feature == 'chol' and value < 200:
                    risk_score = max(0, risk_score - 25)  # Lower risk for normal cholesterol
                
                risk_scores.append(risk_score)
                feature_labels.append(feature_names.get(feature, feature.replace('_', ' ').title()))
        
        if not risk_scores:
            print("No risk scores calculated. Cannot create key risk factors chart.")
            return None
        
        # Create the horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bars with LIGHT BLUE color
        bars = ax.barh(range(len(feature_labels)), risk_scores, color='#87CEEB', alpha=0.8, height=0.6)
        
        # Customize the plot
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=10)
        ax.set_xlabel('Risk Score (0-100)', fontsize=12)
        ax.set_title('Your Key Risk Factors', fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis limits and add grid
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels on the bars
        for i, (bar, score) in enumerate(zip(bars, risk_scores)):
            ax.text(score + 2, i, f'{score:.0f}', va='center', fontsize=9, fontweight='bold')
        
        # Add a vertical line at 50 for reference
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout(pad=1.5)
        return fig
        
    except Exception as e:
        print(f"ERROR in create_key_risk_factors_chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_patient_risk_factor_plot(input_data, feature_importance):
    """
    Create a plot showing the patient's risk factors.
    
    Parameters:
    - input_data: Patient's data
    - feature_importance: DataFrame with feature importances
    
    Returns:
    - fig: Matplotlib figure
    """
    # Clear any existing plots
    plt.clf()
    
    print(f"DEBUG: create_patient_risk_factor_plot called with input_data keys: {list(input_data.keys())}")

    # Ensure feature_importance is a DataFrame
    if not isinstance(feature_importance, pd.DataFrame):
        print(f"Warning: feature_importance is not a DataFrame, got {type(feature_importance)}. Cannot create patient risk factor plot.")
        return None
    
    # Ensure required columns exist
    required_columns = ['Feature', 'Importance']
    if not all(col in feature_importance.columns for col in required_columns):
        print(f"Warning: feature_importance missing required columns. Found: {feature_importance.columns}. Cannot create patient risk factor plot.")
        return None

    # Get the top 5 features
    top_features = feature_importance.head(5)['Feature'].values
    print(f"DEBUG: Top 5 features for risk factor plot: {top_features}")
    
    # Get the patient's values for these features
    patient_values = [input_data[feature] for feature in top_features]
    print(f"DEBUG: Patient values: {patient_values}")
    
    # Normalize the values to 0-100 range for display
    # This is simplified normalization - in practice, you would use domain knowledge
    # or distribution of the training data to define proper scales
    feature_scales = {
        'age': (25, 80),
        'trestbps': (90, 200),
        'chol': (100, 400),
        'thalach': (70, 200),
        'oldpeak': (0, 6),
        'ca': (0, 3),
        'thal': (1, 3),
        'cp': (0, 3),
        'sex': (0, 1),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'exang': (0, 1),
        'slope': (0, 2)
    }
    
    normalized_values = []
    for i, feature in enumerate(top_features):
        if feature in feature_scales:
            min_val, max_val = feature_scales[feature]
            val = input_data[feature]
            norm_val = min(100, max(0, ((val - min_val) / (max_val - min_val)) * 100))
            normalized_values.append(norm_val)
        else:
            # For binary or categorical features, use the value directly (0-100)
            normalized_values.append(input_data[feature] * 100)
    
    print(f"DEBUG: Normalized values: {normalized_values}")
    
    # Create the plot with proper figure handling
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features, normalized_values, color='skyblue', alpha=0.8, height=0.6)
    
    # Add a risk threshold line
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add labels and styling
    ax.set_xlabel('Risk Score (0-100)', fontsize=12)
    ax.set_ylabel('Risk Factors', fontsize=12)
    ax.set_title('Risk Factor Contributions', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    
    # Add grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels to the bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                 f"{normalized_values[i]:.0f}", 
                 va='center', fontsize=9, fontweight='bold')
    
    # Add reference line at 50
    ax.axvline(x=50, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    print(f"DEBUG: Risk factor plot created successfully with enhanced styling")
    return fig

def create_model_consensus_chart(individual_predictions):
    """
    Create a bar chart showing the consensus of individual model predictions.
    
    Parameters:
    - individual_predictions: Dictionary of individual model predictions
    
    Returns:
    - fig: Matplotlib figure
    """
    # Clear any existing plots
    plt.clf()

    if not individual_predictions:
        print("Warning: No individual predictions provided for model consensus chart. Cannot create chart.")
        return None

    model_names = []
    probabilities = []
    for model_name, data in individual_predictions.items():
        if 'probability' in data:
            model_names.append(model_name.replace('_', ' ').title())
            probabilities.append(data['probability'] * 100)
    
    if not model_names:
        print("Warning: No probabilities found in individual predictions. Cannot create chart.")
        return None

    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=model_names, y=probabilities, palette='viridis')

    plt.title('Individual Model Predictions', fontsize=16, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Probability of Heart Disease (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on the bars
    for i, p in enumerate(probabilities):
        ax.text(i, p + 2, f'{p:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def test_risk_factor_chart():
    """Test function to verify chart generation works"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Simple test data
    features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
    values = [75, 60, 85, 45, 90]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, values, color='skyblue')
    ax.set_xlim(0, 100)
    ax.set_title('Test Risk Factor Chart')
    ax.set_xlabel('Test Values')
    
    # Add value labels
    for i, bar in enumerate(bars):
        ax.text(values[i] + 2, bar.get_y() + bar.get_height()/2, 
                f'{values[i]}', va='center')
    
    plt.tight_layout()
    return fig

def convert_plot_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string for HTML embedding
    """
    if fig is None:
        print("DEBUG: convert_plot_to_base64 received None figure")
        return None
    
    print(f"DEBUG: convert_plot_to_base64 called with figure size: {fig.get_size_inches()}")
    print(f"DEBUG: Figure DPI: {fig.dpi}")
    
    try:
        img_data = BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        img_data.seek(0)
        
        # Get the size of the image data
        img_size = len(img_data.getvalue())
        print(f"DEBUG: Image data size: {img_size} bytes")
        
        encoded_img = base64.b64encode(img_data.getvalue()).decode('utf-8')
        
        print(f"DEBUG: Base64 encoded image length: {len(encoded_img)}")
        print(f"DEBUG: Base64 starts with: {encoded_img[:50]}...")
        
        plt.close(fig)  # Close the figure to free up memory
        return encoded_img
    except Exception as e:
        print(f"ERROR in convert_plot_to_base64: {str(e)}")
        plt.close(fig)
        return None

def generate_knn_explanation(input_data, prediction, neighbors_data, feature_importance):
    """
    Generate an explanation specific to the KNN model.
    
    Parameters:
    - input_data: Dictionary with user input values
    - prediction: Prediction result dictionary
    - neighbors_data: DataFrame with neighbor data (can be None)
    - feature_importance: DataFrame with feature importances (can be None)
    
    Returns:
    - explanation: Dictionary with KNN-specific explanations
    """
    # Load the KNN model to get n_neighbors
    model_path = os.path.join('models', 'knn_heart_disease_model.pkl')
    
    # Check if model exists, if not, return empty explanation
    if not os.path.exists(model_path):
        return {
            'model_type': 'knn',
            'model_name': 'K-Nearest Neighbors',
            'explanation_text': 'KNN model not found or not trained. Cannot provide detailed explanation.',
            'top_similar_neighbors': [],
            'comparative_analysis': 'No comparative analysis available without a trained KNN model.'
        }

    model = joblib.load(model_path)
    n_neighbors = model.n_neighbors
    
    similarity_values = []
    positive_neighbors = 0
    similar_cases_percentage = 0.0
    total_neighbors = 0
    neighbors_list = []
    avg_similarity = 0.0

    if neighbors_data is not None and not neighbors_data.empty:
        # Calculate similarity statistics
        # Ensure 'Similarity' column is numeric before conversion
        neighbors_data['Similarity'] = pd.to_numeric(neighbors_data['Similarity'], errors='coerce')
        # Drop rows where 'Similarity' became NaN due to coercion errors
        neighbors_data.dropna(subset=['Similarity'], inplace=True)

        similarity_values = [float(neighbor['Similarity']) for neighbor in neighbors_data.to_dict('records')]
        avg_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0
        
        # Calculate similar cases with heart disease
        positive_neighbors = sum(1 for neighbor in neighbors_data.to_dict('records') if neighbor.get('Prediction') == 'Positive') # Use .get for safety
        total_neighbors = len(neighbors_data)
        similar_cases_percentage = positive_neighbors / total_neighbors * 100 if total_neighbors > 0 else 0.0

        # Add top 5 most similar neighbors for detailed analysis
        neighbors_list = neighbors_data.to_dict('records')
        neighbors_list.sort(key=lambda x: float(x.get('Similarity', 0)), reverse=True) # Use .get for safety
        neighbors_list = [
            {
                'PatientID': f"Patient {i+1}", # Generate a generic patient ID
                'Prediction': n.get('Prediction', 'Unknown'),
                'Probability': n.get('Probability', 0.0),
                'Similarity': n.get('Similarity', 0.0)
            } for i, n in enumerate(neighbors_list[:5])
        ]

    # Create the explanation
    explanation = {
        'model_type': 'knn',
        'model_name': 'K-Nearest Neighbors',
        'n_neighbors': n_neighbors,
        'avg_similarity': avg_similarity,
        'similar_cases_percentage': f"{similar_cases_percentage:.1f}%",
        'similar_cases_count': positive_neighbors,
        'total_neighbors': total_neighbors,
        'explanation_text': f"The KNN model found that {positive_neighbors} out of {total_neighbors} similar patients ({similar_cases_percentage:.1f}%) have heart disease. Your health profile has an average similarity of {avg_similarity:.2f} with these patients." if total_neighbors > 0 else "No sufficient similar cases found to generate a detailed KNN explanation.",
        'top_similar_neighbors': neighbors_list,
        'comparative_analysis': 'Comparative analysis is based on your similarity to other patients.'
    }
    
    return explanation

def generate_random_forest_explanation(input_data, prediction, feature_importance):
    """Generate explanation using Random Forest model insights"""
    try:
        # Debug prints
        print('DEBUG: RF - feature_importance is None?', feature_importance is None)
        print('DEBUG: RF - prediction type:', type(prediction))
        print('DEBUG: RF - prediction value:', prediction)
        
        if feature_importance is not None:
            print('DEBUG: RF - feature_importance columns:', feature_importance.columns)
            print('DEBUG: RF - feature_importance shape:', feature_importance.shape)

        # Ensure prediction is a dictionary
        if isinstance(prediction, str):
            print('DEBUG: RF - prediction is string, creating default dict')
            prediction = {'risk_level': 'Unknown', 'probability': 0.0}
        elif not isinstance(prediction, dict):
            print('DEBUG: RF - prediction is not dict, creating default dict')
            prediction = {'risk_level': 'Unknown', 'probability': 0.0}

        # Get risk level and probability
        risk_level = prediction.get('risk_level', 'Unknown')
        probability = prediction.get('probability', 0.0)
        
        print(f'DEBUG: RF - risk_level: {risk_level}, probability: {probability}')
        
        # Initialize key factors dictionary
        key_factors = {}
        
        # Process feature importance if available
        if feature_importance is not None and not feature_importance.empty:
            # Get top 5 most important features
            top_features = feature_importance.nlargest(5, 'Importance')
            
            # Create key factors dictionary
            for _, row in top_features.iterrows():
                feature = row['Feature']
                importance = row['Importance']
                value = input_data.get(feature, 'N/A')
                
                # Format the value based on feature type
                if feature in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
                    value_str = f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
                elif feature == 'sex':
                    value_str = "Male" if value == 1 else "Female"
                elif feature == 'cp':
                    value_str = {
                        0: "Typical Angina",
                        1: "Atypical Angina",
                        2: "Non-anginal Pain",
                        3: "Asymptomatic"
                    }.get(value, "Unknown")
                else:
                    value_str = str(value)
                
                key_factors[feature] = {
                    'value': value_str,
                    'importance': f"{importance:.2%}",
                    'description': get_feature_description(feature)
                }
        
        # Generate explanation text
        explanation_text = f"""Random Forest Analysis:
Risk Level: {risk_level}
Probability: {probability:.1%}

Key Risk Factors:
{format_key_factors(key_factors)}"""

        # Generate recommendations based on risk level
        recommendations = generate_recommendations(risk_level, key_factors)
        
        result = {
            'explanation_text': explanation_text,
            'key_factors': key_factors,
            'recommendations': recommendations,
            'feature_contributions': feature_importance.to_dict('records') if feature_importance is not None else []
        }
        
        print('DEBUG: RF - successfully created explanation')
        return result
        
    except Exception as e:
        print(f"Error in generate_random_forest_explanation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'explanation_text': f"Random Forest analysis not available due to an error: {str(e)}",
            'key_factors': {},
            'recommendations': {
                'immediate': [],
                'lifestyle': [],
                'medical': [],
                'preventive': []
            },
            'feature_contributions': []
        }

def generate_xgboost_explanation(input_data, prediction, feature_importance):
    """
    Generate an explanation specific to the XGBoost model.
    
    Parameters:
    - input_data: Dictionary with user input values
    - prediction: Prediction result dictionary
    - feature_importance: DataFrame with feature importances for XGBoost
    
    Returns:
    - explanation: Dictionary with XGBoost-specific explanations
    """
    # Load the XGBoost model
    model_path = os.path.join('models', 'xgboost_model.pkl')
    
    # Check if model exists, if not, return empty explanation
    if not os.path.exists(model_path):
        return {
            'model_type': 'xgboost',
            'model_name': 'XGBoost',
            'explanation_text': 'XGBoost model not found or not trained. Cannot provide detailed explanation.',
            'feature_contributions': {}
        }

    model = joblib.load(model_path)
    
    # Get model parameters
    n_trees = model.n_estimators
    learning_rate = model.learning_rate
    
    feature_contributions = {}
    if feature_importance is not None and not feature_importance.empty:
        # Get the top contributing features
        top_features = feature_importance.head(5)['Feature'].values
        
        for feature in top_features:
            value = input_data.get(feature) # Use .get() for safety
            if value is None: # Skip if feature value is not in input_data
                continue

            importance = float(feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0])
            
            # Create feature risk explanation based on value and importance
            if importance > 0.1:  # High importance feature
                if feature == 'cp' and value >= 2:
                    risk = "high"
                    desc = f"Chest pain type ({value:.0f}) indicates significant risk. This is a strong predictor."
                elif feature == 'thal' and value > 2:
                    risk = "high"
                    desc = f"Thalassemia value ({value:.0f}) is abnormal, a critical factor in XGBoost's prediction."
                elif feature == 'ca' and value > 0:
                    risk = "high"
                    desc = f"Number of major vessels ({value:.0f}) is a key factor in your risk assessment."
                elif feature == 'exang' and value == 1:
                    risk = "high"
                    desc = f"Exercise induced angina detected, significantly increasing risk prediction."
                else:
                    # Generic high importance
                    risk = "medium"
                    desc = f"{feature.replace('_', ' ').title()} ({value:.1f}) is an important factor in the XGBoost model's prediction."
            else:
                # Lower importance feature
                risk = "low"
                desc = f"{feature.replace('_', ' ').title()} ({value:.1f}) has a minor influence on the prediction."
            
            feature_contributions[feature] = {
                'value': value,
                'importance': importance,
                'risk_level': risk,
                'description': desc
            }
    
    # Create the explanation
    explanation = {
        'model_type': 'xgboost',
        'model_name': 'XGBoost',
        'n_trees': n_trees,
        'learning_rate': learning_rate,
        'feature_contributions': feature_contributions,
        'explanation_text': f"The XGBoost model, our most advanced algorithm, used {n_trees} boosted trees to analyze complex patterns in your data. XGBoost excels at identifying subtle interactions between risk factors." if feature_contributions else "No sufficient feature importance data to generate a detailed XGBoost explanation."
    }
    
    return explanation

def get_model_specific_feature_importance(model_dir='models', model_type=None):
    """
    Get feature importance for a specific model type.
    
    Parameters:
    - model_dir: Directory where models are stored
    - model_type: Type of model ('knn', 'random_forest', 'xgboost')
    
    Returns:
    - importance_df: DataFrame with feature importances
    """
    # Determine which model to use
    if model_type == 'knn':
        model_path = os.path.join(model_dir, 'knn_model.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'knn_heart_disease_model.pkl')
    elif model_type == 'random_forest':
        model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    elif model_type == 'xgboost':
        model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    else:
        # Default to KNN if model type is not recognized
        model_path = os.path.join(model_dir, 'knn_heart_disease_model.pkl')
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Returning empty importance DataFrame.")
        return pd.DataFrame({'Feature': [], 'Importance': [], 'StdDev': []})

    # Load the model
    model = joblib.load(model_path)
    
    # Load the feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    if not os.path.exists(feature_names_path):
        print(f"Warning: Feature names not found at {feature_names_path}. Returning empty importance DataFrame.")
        return pd.DataFrame({'Feature': [], 'Importance': [], 'StdDev': []})
    feature_names = joblib.load(feature_names_path)
    
    # Load training data
    data_path = os.path.join('data', 'heart_disease.csv')
    if not os.path.exists(data_path):
        print(f"Warning: Training data not found at {data_path}. Returning empty importance DataFrame.")
        return pd.DataFrame({'Feature': [], 'Importance': [], 'StdDev': []})
    df = pd.read_csv(data_path)
    
    # Prepare data
    X_train = df.drop('target', axis=1).values
    y_train = df['target'].values
    
    # If the model has built-in feature importance (RF or XGBoost), use it
    if model_type in ['random_forest', 'xgboost'] and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        std = np.zeros_like(importances)  # No std for built-in importance
    else:
        # For KNN or if built-in importance isn't available, use permutation importance
        try:
            result = permutation_importance(
                model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
            std = result.importances_std
        except Exception as e:
            print(f"Error calculating permutation importance: {str(e)}. Returning empty importance DataFrame.")
            return pd.DataFrame({'Feature': [], 'Importance': [], 'StdDev': []})
    
    # Create a DataFrame with the feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'StdDev': std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def generate_explanation(input_data, prediction, neighbors_data, feature_importance, individual_predictions=None):
    """Generate a comprehensive explanation of the prediction"""
    try:
        # Debug prints for diagnosis
        print('DEBUG: neighbors_data is None?', neighbors_data is None)
        if neighbors_data is not None:
            print('DEBUG: neighbors_data columns:', neighbors_data.columns)
            print('DEBUG: neighbors_data shape:', neighbors_data.shape)
        print('DEBUG: input_data keys:', list(input_data.keys()))
        print('DEBUG: feature_importance is None?', feature_importance is None)
        if feature_importance is not None:
            print('DEBUG: feature_importance columns:', feature_importance.columns)
            print('DEBUG: feature_importance shape:', feature_importance.shape)

        # Get risk level and probability
        risk_level = prediction.get('risk_level', 'Unknown')
        probability = prediction.get('probability', 0.0)
        risk_type = prediction.get('risk_type', 'Unknown')
        risk_description = prediction.get('risk_description', '')
        
        # Generate model-specific explanations
        knn_explanation = generate_knn_explanation(input_data, prediction, neighbors_data, feature_importance)
        rf_explanation = generate_random_forest_explanation(input_data, prediction, feature_importance)
        xgb_explanation = generate_xgboost_explanation(input_data, prediction, feature_importance)
        
        # Create visualizations with better error handling
        print("DEBUG: Creating visualizations...")
        print(f"DEBUG: feature_importance type: {type(feature_importance)}")
        if feature_importance is not None:
            print(f"DEBUG: feature_importance shape: {feature_importance.shape if hasattr(feature_importance, 'shape') else 'No shape'}")
            print(f"DEBUG: feature_importance columns: {feature_importance.columns if hasattr(feature_importance, 'columns') else 'No columns'}")
            print(f"DEBUG: feature_importance head: {feature_importance.head() if hasattr(feature_importance, 'head') else 'No head method'}")
        else:
            print("DEBUG: feature_importance is None")
        
        feature_importance_plot = None
        feature_comparison_plot = None
        risk_factor_plot = None
        key_risk_factors_chart = None
        model_consensus_chart = None
        
        try:
            feature_importance_plot = create_feature_importance_plot(feature_importance)
            print(f"DEBUG: feature_importance_plot created: {feature_importance_plot is not None}")
        except Exception as e:
            print(f"ERROR creating feature_importance_plot: {str(e)}")
            feature_importance_plot = None
            
        # Fallback: Create a simple feature importance plot if the main one failed
        if feature_importance_plot is None and feature_importance is not None:
            try:
                print("DEBUG: Attempting to create fallback feature importance plot...")
                # Create a simple bar chart using matplotlib
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Get top 5 features
                if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
                    top_features = feature_importance.head(5)
                    features = top_features['Feature'].tolist()
                    importances = top_features['Importance'].tolist()
                    
                    # Create simple horizontal bar chart
                    y_pos = range(len(features))
                    bars = ax.barh(y_pos, importances, color='lightblue', alpha=0.8)
                    
                    # Add labels
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(features, fontsize=10)
                    ax.set_xlabel('Importance', fontsize=12)
                    ax.set_title('Top Feature Importance', fontsize=14, fontweight='bold')
                    
                    # Add value labels
                    for i, (bar, importance) in enumerate(zip(bars, importances)):
                        ax.text(importance + 0.01, i, f'{importance:.3f}', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    feature_importance_plot = fig
                    print("DEBUG: Fallback feature importance plot created successfully")
                else:
                    print("DEBUG: feature_importance data is not suitable for fallback plot")
            except Exception as e:
                print(f"ERROR creating fallback feature_importance_plot: {str(e)}")
                feature_importance_plot = None
        
        try:
            feature_comparison_plot = create_feature_comparison_plot(neighbors_data, input_data)
            print(f"DEBUG: feature_comparison_plot created: {feature_comparison_plot is not None}")
        except Exception as e:
            print(f"ERROR creating feature_comparison_plot: {str(e)}")
            feature_comparison_plot = None
        
        try:
            risk_factor_plot = create_patient_risk_factor_plot(input_data, feature_importance)
            print(f"DEBUG: risk_factor_plot created: {risk_factor_plot is not None}")
        except Exception as e:
            print(f"ERROR creating risk_factor_plot: {str(e)}")
            risk_factor_plot = None
        
        try:
            key_risk_factors_chart = create_key_risk_factors_chart(input_data, feature_importance)
            print(f"DEBUG: key_risk_factors_chart created: {key_risk_factors_chart is not None}")
        except Exception as e:
            print(f"ERROR creating key_risk_factors_chart: {str(e)}")
            key_risk_factors_chart = None
        
        try:
            model_consensus_chart = create_model_consensus_chart(individual_predictions) if individual_predictions else None
            print(f"DEBUG: model_consensus_chart created: {model_consensus_chart is not None}")
        except Exception as e:
            print(f"ERROR creating model_consensus_chart: {str(e)}")
            model_consensus_chart = None
        
        # Convert plots to base64 with error handling
        print("DEBUG: Converting plots to base64...")
        
        feature_importance_img = None
        feature_comparison_img = None
        risk_factor_img = None
        key_risk_factors_img = None
        model_consensus_img = None
        
        try:
            feature_importance_img = convert_plot_to_base64(feature_importance_plot)
            print(f"DEBUG: feature_importance_img created: {feature_importance_img is not None}")
        except Exception as e:
            print(f"ERROR converting feature_importance_plot to base64: {str(e)}")
            feature_importance_img = None
        
        try:
            feature_comparison_img = convert_plot_to_base64(feature_comparison_plot)
            print(f"DEBUG: feature_comparison_img created: {feature_comparison_img is not None}")
        except Exception as e:
            print(f"ERROR converting feature_comparison_plot to base64: {str(e)}")
            feature_comparison_img = None
        
        try:
            risk_factor_img = convert_plot_to_base64(risk_factor_plot)
            print(f"DEBUG: risk_factor_img created: {risk_factor_img is not None}")
        except Exception as e:
            print(f"ERROR converting risk_factor_plot to base64: {str(e)}")
            risk_factor_img = None
        
        try:
            key_risk_factors_img = convert_plot_to_base64(key_risk_factors_chart)
            print(f"DEBUG: key_risk_factors_img created: {key_risk_factors_img is not None}")
        except Exception as e:
            print(f"ERROR converting key_risk_factors_chart to base64: {str(e)}")
            key_risk_factors_img = None
        
        try:
            model_consensus_img = convert_plot_to_base64(model_consensus_chart) if model_consensus_chart else None
            print(f"DEBUG: model_consensus_img created: {model_consensus_img is not None}")
        except Exception as e:
            print(f"ERROR converting model_consensus_chart to base64: {str(e)}")
            model_consensus_img = None
        
        # --- Boxplots for Similar Cases ---
        boxplot_features = ['age', 'sex', 'cp', 'trestbps', 'chol']
        boxplots = {}
        if neighbors_data is not None:
            for feature in boxplot_features:
                if feature in neighbors_data.columns and feature in input_data:
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.boxplot(neighbors_data[feature], vert=False, widths=0.7)
                    ax.scatter(input_data[feature], 1, color='red', s=100, zorder=10)
                    ax.set_title(f'Comparison for {feature}')
                    ax.set_xlabel('Value')
                    ax.set_yticks([])
                    ax.legend(['Your value', 'Neighbors'], loc='upper right')
                    plt.tight_layout()
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                    boxplots[feature] = img_base64
        print('DEBUG: boxplots keys:', list(boxplots.keys()))
        print('DEBUG: key_factors keys:', list(rf_explanation.get('key_factors', {}).keys()))
        
        # Generate comprehensive explanation text
        explanation_text = f"""Heart Disease Risk Assessment Results

Risk Level: {risk_level}
Probability of Heart Disease: {probability:.1%}
Risk Type: {risk_type}

Detailed Analysis:
{risk_description}

Key Risk Factors Analysis:
{rf_explanation.get('key_factors', {})}

Model-Specific Insights:
1. K-Nearest Neighbors Analysis:
{knn_explanation.get('explanation_text', '')}

2. Random Forest Analysis:
{rf_explanation.get('explanation_text', '')}

3. XGBoost Analysis:
{xgb_explanation.get('explanation_text', '')}

Comparative Analysis:
{knn_explanation.get('comparative_analysis', '')}

Recommendations:
1. Immediate Actions:
{rf_explanation.get('recommendations', {}).get('immediate', [])}

2. Lifestyle Changes:
{rf_explanation.get('recommendations', {}).get('lifestyle', [])}

3. Medical Follow-up:
{rf_explanation.get('recommendations', {}).get('medical', [])}

4. Preventive Measures:
{rf_explanation.get('recommendations', {}).get('preventive', [])}

Note: This assessment is based on multiple machine learning models and should be used as a guide. Please consult with healthcare professionals for medical advice."""

        # Create the explanation dictionary
        explanation = {
            'risk_level': risk_level,
            'risk_type': risk_type,
            'probability': probability,
            'risk_description': risk_description,
            'explanation_text': explanation_text,
            'feature_importance_img': feature_importance_img,
            'feature_comparison_img': feature_comparison_img,
            'risk_factor_img': risk_factor_img,
            'key_risk_factors_img': key_risk_factors_img,
            'model_consensus_img': model_consensus_img,
            'individual_predictions': individual_predictions if individual_predictions is not None else {},
            'key_factors': rf_explanation.get('key_factors', {}),
            'recommendations': rf_explanation.get('recommendations', {
                'immediate': [],
                'lifestyle': [],
                'medical': [],
                'preventive': []
            }),
            'model_insights': {
                'knn': knn_explanation,
                'random_forest': rf_explanation,
                'xgboost': xgb_explanation
            },
            'boxplots': boxplots
        }
        
        return explanation
        
    except Exception as e:
        print(f"Error in generate_explanation: {str(e)}")
        # Return a basic explanation if there's an error, ensuring all expected keys are present
        return {
            'risk_level': prediction.get('risk_level', 'Unknown'),
            'risk_type': prediction.get('risk_type', 'Unknown'),
            'probability': prediction.get('probability', 0.0),
            'risk_description': prediction.get('risk_description', ''),
            'explanation_text': f"Risk Level: {prediction.get('risk_level', 'Unknown')}\nProbability: {prediction.get('probability', 0.0):.1%}\n\n{prediction.get('risk_description', '')}\n\nAn error occurred while generating a detailed explanation: {str(e)}",
            'feature_importance_img': None,
            'feature_comparison_img': None,
            'risk_factor_img': None,
            'key_risk_factors_img': None,
            'model_consensus_img': None,
            'individual_predictions': individual_predictions if individual_predictions is not None else {},
            'key_factors': {},
            'recommendations': {
                'immediate': [],
                'lifestyle': [],
                'medical': [],
                'preventive': []
            },
            'model_insights': {
                'knn': {
                    'explanation_text': 'KNN explanation not available due to an error.',
                    'top_similar_neighbors': [],
                    'comparative_analysis': 'No comparative analysis available due to an error.'
                },
                'random_forest': {
                    'explanation_text': 'Random Forest explanation not available due to an error.',
                    'key_factors': {},
                    'recommendations': {
                        'immediate': [], 'lifestyle': [], 'medical': [], 'preventive': []
                    }
                },
                'xgboost': {
                    'explanation_text': 'XGBoost explanation not available due to an error.',
                    'feature_contributions': {}
                }
            },
            'boxplots': {}
        } 

def format_key_factors(key_factors):
    """Format key factors for display in explanations"""
    if not key_factors:
        return "No key risk factors identified."
    
    formatted_text = []
    for feature, data in key_factors.items():
        formatted_text.append(f"- {feature.title()}: {data['value']} (Importance: {data['importance']})")
        if 'description' in data:
            formatted_text.append(f"  {data['description']}")
    
    return "\n".join(formatted_text)

def get_feature_description(feature):
    """Get a human-readable description of a feature"""
    descriptions = {
        'age': 'Age in years',
        'sex': 'Gender (1 = Male, 0 = Female)',
        'cp': 'Chest pain type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)',
        'restecg': 'Resting electrocardiographic results',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment',
        'ca': 'Number of major vessels colored by fluoroscopy',
        'thal': 'Thalassemia (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)'
    }
    return descriptions.get(feature, 'No description available')

def generate_recommendations(risk_level, key_factors):
    """Generate personalized recommendations based on risk level and key factors"""
    recommendations = {
        'immediate': [],
        'lifestyle': [],
        'medical': [],
        'preventive': []
    }
    
    # Immediate actions based on risk level
    if risk_level == 'High':
        recommendations['immediate'].extend([
            "Schedule an immediate appointment with your healthcare provider",
            "Monitor your blood pressure and heart rate daily",
            "Keep a symptom diary to track any chest pain or discomfort"
        ])
    elif risk_level == 'Moderate':
        recommendations['immediate'].extend([
            "Schedule a follow-up with your healthcare provider within a week",
            "Start monitoring your vital signs regularly",
            "Review your current medications with your doctor"
        ])
    else:  # Low risk
        recommendations['immediate'].extend([
            "Schedule a routine check-up with your healthcare provider",
            "Continue monitoring your health regularly",
            "Maintain your current healthy habits"
        ])
    
    # Lifestyle recommendations based on key factors
    if 'age' in key_factors:
        age = float(key_factors['age']['value'])
        if age > 50:
            recommendations['lifestyle'].append("Consider age-appropriate exercise routines")
    
    if 'trestbps' in key_factors:
        bp = float(key_factors['trestbps']['value'])
        if bp > 120:
            recommendations['lifestyle'].extend([
                "Reduce sodium intake",
                "Practice stress management techniques",
                "Consider DASH diet for blood pressure management"
            ])
    
    if 'chol' in key_factors:
        chol = float(key_factors['chol']['value'])
        if chol > 200:
            recommendations['lifestyle'].extend([
                "Reduce saturated fat intake",
                "Increase fiber consumption",
                "Consider Mediterranean diet"
            ])
    
    if 'thalach' in key_factors:
        thalach = float(key_factors['thalach']['value'])
        if thalach < 150:
            recommendations['lifestyle'].append("Start a gradual exercise program to improve heart rate")
    
    # Medical recommendations
    if risk_level in ['High', 'Moderate']:
        recommendations['medical'].extend([
            "Regular ECG monitoring",
            "Consider stress test evaluation",
            "Review and possibly adjust current medications"
        ])
    
    # Preventive measures
    recommendations['preventive'].extend([
        "Regular health check-ups",
        "Maintain a healthy weight",
        "Stay physically active",
        "Manage stress levels",
        "Get adequate sleep"
    ])
    
    return recommendations 

def create_feature_impact_chart(input_data, feature_importance):
    """
    Create a chart showing feature impact analysis for the "Detailed Feature Impact Analysis" section.
    
    Parameters:
    - input_data: Dictionary with user input values
    - feature_importance: DataFrame with feature importances
    
    Returns:
    - fig: Matplotlib figure
    """
    try:
        # Clear all matplotlib state completely
        plt.close('all')
        plt.clf()
        plt.cla()
        
        # Ensure feature_importance is a DataFrame
        if not isinstance(feature_importance, pd.DataFrame):
            print(f"Warning: feature_importance is not a DataFrame, got {type(feature_importance)}. Cannot create feature impact chart.")
            return None
        
        # Get top 6 features by importance
        if 'Feature' in feature_importance.columns and 'Importance' in feature_importance.columns:
            top_features = feature_importance.nlargest(6, 'Importance')
        else:
            print("Warning: Required columns not found in feature_importance")
            return None
        
        # Feature display names
        feature_display_names = {
            'age': 'Age',
            'sex': 'Gender',
            'cp': 'Chest Pain Type',
            'trestbps': 'Blood Pressure',
            'chol': 'Cholesterol',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate',
            'exang': 'Exercise Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels',
            'thal': 'Thalassemia'
        }
        
        # Calculate impact scores for better visualization
        impact_scores = []
        feature_names = []
        raw_scores = []
        
        for _, row in top_features.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            if feature in input_data:
                # Calculate impact score (importance * normalized value)
                value = input_data[feature]
                
                # Normalize value to 0-1 scale based on typical ranges
                if feature == 'age':
                    normalized = min(1.0, max(0.0, (value - 20) / 80))
                elif feature == 'trestbps':
                    normalized = min(1.0, max(0.0, (value - 90) / 110))
                elif feature == 'chol':
                    normalized = min(1.0, max(0.0, (value - 100) / 400))
                elif feature == 'thalach':
                    normalized = min(1.0, max(0.0, (220 - value) / 150))  # Inverse for heart rate
                elif feature == 'oldpeak':
                    normalized = min(1.0, max(0.0, value / 6))
                elif feature in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                    # For categorical features, use direct value scaling
                    max_val = {'sex': 1, 'cp': 3, 'fbs': 1, 'restecg': 2, 'exang': 1, 'slope': 2, 'ca': 3, 'thal': 3}.get(feature, 3)
                    normalized = value / max_val
                else:
                    normalized = 0.5  # Default middle value
                
                # Calculate raw impact score
                raw_impact = importance * normalized * 100
                raw_scores.append(raw_impact)
                feature_names.append(feature_display_names.get(feature, feature.replace('_', ' ').title()))
        
        if not raw_scores:
            print("No impact scores calculated. Cannot create feature impact chart.")
            return None
        
        # Scale the scores for better visualization (normalize to 20-100 range)
        min_raw = min(raw_scores)
        max_raw = max(raw_scores)
        if max_raw > min_raw:
            # Scale to 20-100 range for better visualization
            scaled_scores = [20 + (score - min_raw) / (max_raw - min_raw) * 80 for score in raw_scores]
        else:
            # All scores are the same, set them to 60
            scaled_scores = [60] * len(raw_scores)
        
        impact_scores = scaled_scores
        
        # Create figure with explicit size and higher DPI for better quality
        fig = plt.figure(figsize=(10, 7), dpi=120)
        ax = fig.add_subplot(111)
        
        # Create horizontal bar chart with gradient colors
        y_positions = range(len(feature_names))
        
        # Use a color gradient based on impact scores
        colors = plt.cm.RdYlBu_r([score/100 for score in impact_scores])
        bars = ax.barh(y_positions, impact_scores, color=colors, alpha=0.85, height=0.7, edgecolor='white', linewidth=1)
        
        # Customize the chart with enhanced styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_names, fontsize=11, fontweight='bold')
        ax.set_xlabel('Impact Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Impact Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, impact_scores)):
            ax.text(score + 1, i, f'{score:.0f}', va='center', ha='left', fontsize=10, fontweight='bold', color='white')
        
        # Add grid and styling
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_xlim(0, max(impact_scores) * 1.1)
        
        # Adjust layout to prevent stretching
        plt.tight_layout(pad=1.5)
        
        return fig
        
    except Exception as e:
        print(f"ERROR in create_feature_impact_chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 