#!/usr/bin/env python3
"""
Startup script for Heart Disease Predictor application.
Handles initialization and startup process for Render deployment.
"""

import os
import sys
import time
import threading
from flask import Flask

def initialize_application():
    """Initialize the application with proper error handling"""
    try:
        print("🚀 Starting Heart Disease Predictor initialization...")
        
        # Import the main app
        from app import app, init_db, User, download_heart_disease_data, train_and_save_models
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('temp', exist_ok=True)
        
        print("✅ Directories created")
        
        # Initialize database
        with app.app_context():
            init_db()
            print("✅ Database initialized")
            
            # Check if any users exist
            try:
                user_count = User.query.count()
                if user_count == 0:
                    print("⚠️  No users found in database")
                    print("💡 Create your first account via the signup page")
                else:
                    print(f"✅ Database has {user_count} user(s)")
            except Exception as e:
                print(f"⚠️  Database check failed: {str(e)}")
        
        # Check if models exist, if not train them
        knn_model_path = os.path.join('models', 'knn_heart_disease_model.pkl')
        rf_model_path = os.path.join('models', 'random_forest_model.pkl')
        xgb_model_path = os.path.join('models', 'xgboost_model.pkl')
        
        models_exist = (os.path.exists(knn_model_path) and 
                       os.path.exists(rf_model_path) and 
                       os.path.exists(xgb_model_path))
        
        if not models_exist:
            print("🤖 Training machine learning models...")
            
            # Check if data exists, if not download it
            data_path = os.path.join('data', 'heart_disease.csv')
            if not os.path.exists(data_path):
                print("📥 Downloading heart disease dataset...")
                download_heart_disease_data()
                print("✅ Dataset downloaded")
            
            # Train all models
            print("🔄 Training models (this may take a few minutes)...")
            train_and_save_models(data_path, 'models')
            print("✅ Models trained successfully!")
        else:
            print("✅ Pre-trained models found")
        
        return app
        
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        print("🔄 Attempting to start with fallback configuration...")
        
        # Return a minimal app if initialization fails
        fallback_app = Flask(__name__)
        fallback_app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-key')
        
        @fallback_app.route('/')
        def fallback_home():
            return """
            <h1>Heart Disease Predictor</h1>
            <p>Application is starting up. Please wait a moment and refresh the page.</p>
            <p>If this message persists, please check the application logs.</p>
            """
        
        return fallback_app

def start_application():
    """Start the application with proper port binding"""
    try:
        # Initialize the application
        app = initialize_application()
        
        # Get port from environment variable (for Render)
        port = int(os.environ.get('PORT', 8080))
        
        print("🚀 Starting Heart Care application...")
        print(f"🌐 Application will be available on port: {port}")
        print("📚 API Documentation: /help")
        
        # Start the application
        app.run(
            host='0.0.0.0',  # Bind to all interfaces
            port=port,        # Use the port from environment
            debug=False,      # Disable debug mode for production
            threaded=True     # Enable threading for better performance
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    start_application() 