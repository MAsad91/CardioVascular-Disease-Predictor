import os
import numpy as np
import joblib
from .preprocess import preprocess_input

class HeartDiseasePredictor:
    def __init__(self, model_dir='models'):
        """
        Initialize the HeartDiseasePredictor.
        
        Parameters:
        - model_dir: Directory where the model and dependencies are stored
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler_path = None
        self.feature_names_path = None
        self._load_model_and_dependencies()

    def _load_model_and_dependencies(self):
        """
        Load the trained model and its dependencies.
        """
        # Load the model
        model_path = os.path.join(self.model_dir, 'knn_heart_disease_model.pkl')
        self.model = joblib.load(model_path)
        
        # Paths to dependencies
        self.scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        self.feature_names_path = os.path.join(self.model_dir, 'feature_names.pkl')

    def predict(self, input_data):
        """
        Predict heart disease risk for a given input.
        
        Parameters:
        - input_data: Dictionary with user input values
        
        Returns:
        - prediction: Dictionary with prediction results
        """
        # Preprocess the input
        processed_input = preprocess_input(input_data, self.scaler_path, self.feature_names_path)
        
        # Make prediction
        probability = self.model.predict_proba(processed_input)[0][1]
        
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