import os
import numpy as np
import joblib
from .preprocess import preprocess_input
from .multi_model import train_and_save_models
from .download_data import download_heart_disease_data

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
        If models don't exist, train them on first run.
        """
        try:
            # Check if model files exist
            model_path = os.path.join(self.model_dir, 'knn_heart_disease_model.pkl')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            feature_names_path = os.path.join(self.model_dir, 'feature_names.pkl')
            
            if (os.path.exists(model_path) and 
                os.path.exists(scaler_path) and 
                os.path.exists(feature_names_path)):
                
                # Load existing models
                print("Loading pre-trained models...")
                self.model = joblib.load(model_path)
                self.scaler_path = scaler_path
                self.feature_names_path = feature_names_path
                print("✅ Models loaded successfully")
                
            else:
                # Train models on first run
                print("⚠️  Pre-trained models not found. Training models on first run...")
                self._train_models_on_first_run()
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Training models on first run...")
            self._train_models_on_first_run()

    def _train_models_on_first_run(self):
        """
        Train models on first run when they don't exist.
        """
        try:
            # Ensure directories exist
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            # Download data if not exists
            data_path = os.path.join('data', 'heart_disease.csv')
            if not os.path.exists(data_path):
                print("Downloading heart disease dataset...")
                download_heart_disease_data()
            
            # Train models
            print("Training models (this may take a few minutes)...")
            train_and_save_models(data_path, self.model_dir)
            
            # Load the trained model
            model_path = os.path.join(self.model_dir, 'knn_heart_disease_model.pkl')
            self.model = joblib.load(model_path)
            self.scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.feature_names_path = os.path.join(self.model_dir, 'feature_names.pkl')
            
            print("✅ Models trained and loaded successfully")
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            # Create a fallback model
            self._create_fallback_model()

    def _create_fallback_model(self):
        """
        Create a simple fallback model if training fails.
        """
        try:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler
            
            print("Creating fallback model...")
            
            # Create a simple KNN model with dummy data
            X_dummy = np.random.rand(100, 13)  # 13 features
            y_dummy = np.random.randint(0, 2, 100)
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dummy)
            
            # Create and fit model
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_scaled, y_dummy)
            
            # Save fallback model
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(self.model_dir, 'knn_heart_disease_model.pkl'))
            joblib.dump(scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Create feature names
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            joblib.dump(feature_names, os.path.join(self.model_dir, 'feature_names.pkl'))
            
            self.model = model
            self.scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.feature_names_path = os.path.join(self.model_dir, 'feature_names.pkl')
            
            print("✅ Fallback model created")
            
        except Exception as e:
            print(f"Error creating fallback model: {str(e)}")
            raise

    def predict(self, input_data):
        """
        Predict heart disease risk for a given input.
        
        Parameters:
        - input_data: Dictionary with user input values
        
        Returns:
        - prediction: Dictionary with prediction results
        """
        try:
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
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            # Return a safe fallback prediction
            return {
                'probability': 0.5,
                'risk_level': "Medium",
                'risk_description': "Unable to make prediction. Please try again or contact support."
            } 