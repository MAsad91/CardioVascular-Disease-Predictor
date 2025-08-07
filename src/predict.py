import os
import numpy as np
import joblib

from .preprocess import preprocess_input

def load_model_and_dependencies(model_dir='../models'):
    """
    Load the trained model and its dependencies.
    
    Parameters:
    - model_dir: Directory where the model and dependencies are stored
    
    Returns:
    - model: Loaded KNN model
    - scaler_path: Path to the scaler
    - feature_names_path: Path to the feature names
    """
    # Load the model
    model_path = os.path.join(model_dir, 'knn_heart_disease_model.pkl')
    model = joblib.load(model_path)
    
    # Paths to dependencies
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    
    return model, scaler_path, feature_names_path

def predict_heart_disease(input_data, model_dir='../models'):
    """
    Predict heart disease risk for a given input.
    
    Parameters:
    - input_data: Dictionary with user input values
    - model_dir: Directory where the model and dependencies are stored
    
    Returns:
    - prediction: Dictionary with prediction results
    """
    # Load the model and dependencies
    model, scaler_path, feature_names_path = load_model_and_dependencies(model_dir)
    
    # Preprocess the input
    processed_input = preprocess_input(input_data, scaler_path, feature_names_path)
    
    # Make prediction
    probability = model.predict_proba(processed_input)[0][1]
    
    # Determine risk category
    if probability >= 0.7:
        risk_level = "High"
        risk_description = "High risk of heart disease. Immediate medical consultation is recommended."
    elif probability >= 0.4:
        risk_level = "Medium"
        risk_description = "Medium risk of heart disease. Regular check-ups advised."
    else:
        risk_level = "Low"
        risk_description = "Low risk of heart disease. Maintain a healthy lifestyle."
    
    # Create prediction result
    prediction = {
        'probability': float(probability),
        'risk_level': risk_level,
        'risk_description': risk_description
    }
    
    return prediction 

def predict_quick_assessment(input_data, model_dir='models'):
    """
    Predict heart disease risk for quick assessment using the dedicated quick assessment model.
    This model is trained specifically on the essential features used in quick assessment.
    """
    try:
        # Load the quick assessment model and dependencies
        model_path = os.path.join(model_dir, 'quick_assessment_model.pkl')
        scaler_path = os.path.join(model_dir, 'quick_assessment_scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'quick_assessment_features.pkl')
        
        # Check if quick assessment model exists
        if not os.path.exists(model_path):
            print("Quick assessment model not found. Using fallback method...")
            return predict_quick_assessment_fallback(input_data, model_dir)
        
        # Load the model and dependencies
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        quick_features = joblib.load(feature_names_path)
        
        # Prepare input data with only the quick assessment features
        quick_input = {}
        for feature in quick_features:
            if feature in input_data:
                quick_input[feature] = input_data[feature]
            else:
                # Use reasonable defaults for missing features
                if feature == 'restecg':
                    quick_input[feature] = 0  # Default: normal
                else:
                    quick_input[feature] = 0
        
        # Convert to array in the correct order
        X = np.array([[quick_input[feature] for feature in quick_features]])
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Determine risk category
        if probability >= 0.6:  # Lowered threshold from 0.7 to 0.6
            risk_level = "High"
            risk_description = "High risk of heart disease. Immediate medical consultation is recommended."
        elif probability >= 0.3:  # Lowered threshold from 0.4 to 0.3
            risk_level = "Medium"
            risk_description = "Medium risk of heart disease. Regular check-ups advised."
        else:
            risk_level = "Low"
            risk_description = "Low risk of heart disease. Maintain a healthy lifestyle."
        
        # Create prediction result
        prediction = {
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_description': risk_description
        }
        
        return prediction
        
    except Exception as e:
        print(f"Error in quick assessment prediction: {str(e)}")
        # Fallback to the old method if there's an error
        return predict_quick_assessment_fallback(input_data, model_dir)

def predict_quick_assessment_fallback(input_data, model_dir='models'):
    """
    Fallback method for quick assessment using the full model with defaults.
    This is used when the quick assessment model is not available.
    """
    try:
        # Load the existing model and dependencies
        model, scaler_path, feature_names_path = load_model_and_dependencies(model_dir)
        
        # Create a complete input data with defaults for missing features
        complete_input = {
            'age': input_data.get('age', 50),
            'sex': input_data.get('sex', 1),
            'cp': 0,  # Default: typical angina
            'trestbps': input_data.get('trestbps', 120),
            'chol': input_data.get('chol', 200),
            'fbs': 0,  # Default: fasting blood sugar <= 120 mg/dl
            'restecg': input_data.get('restecg', 0),  # Default: normal
            'thalach': input_data.get('thalach', 150),
            'exang': 0,  # Default: no exercise induced angina
            'oldpeak': input_data.get('oldpeak', 0.0),
            'slope': 0,  # Default: upsloping
            'ca': 0,  # Default: 0 major vessels
            'thal': 1  # Default: normal
        }
        
        # Update with provided values
        complete_input.update(input_data)
        
        # Preprocess the input
        processed_input = preprocess_input(complete_input, scaler_path, feature_names_path)
        
        # Make prediction
        probability = model.predict_proba(processed_input)[0][1]
        
        # Determine risk category
        if probability >= 0.7:
            risk_level = "High"
            risk_description = "High risk of heart disease. Immediate medical consultation is recommended."
        elif probability >= 0.4:
            risk_level = "Medium"
            risk_description = "Medium risk of heart disease. Regular check-ups advised."
        else:
            risk_level = "Low"
            risk_description = "Low risk of heart disease. Maintain a healthy lifestyle."
        
        # Create prediction result
        prediction = {
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_description': risk_description
        }
        
        return prediction
        
    except Exception as e:
        print(f"Error in quick assessment fallback prediction: {str(e)}")
        # Return a default prediction in case of error
        return {
            'probability': 0.3,
            'risk_level': 'Medium',
            'risk_description': 'Unable to complete assessment. Please try the full assessment for more accurate results.'
        } 