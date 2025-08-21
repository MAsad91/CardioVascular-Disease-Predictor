import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # MongoDB Atlas configuration (for future use)
    MONGODB_SETTINGS = {
        'db': 'heart_disease_db',
        'host': os.getenv('MONGODB_URI', 'mongodb+srv://mudassirshahid605:YOUR_ACTUAL_PASSWORD_HERE@cluster0.dhmfcnv.mongodb.net/heart_disease_db?retryWrites=true&w=majority&appName=Cluster0'),
        'connect': True,
        'retryWrites': True,
        'w': 'majority',
        'tls': True,
        'tlsAllowInvalidCertificates': True,
        'tlsInsecure': True,
        'serverSelectionTimeoutMS': 5000,
        'connectTimeoutMS': 10000,
        'socketTimeoutMS': 20000,
        'maxPoolSize': 50,
        'minPoolSize': 10,
        'maxIdleTimeMS': 30000,
        'waitQueueTimeoutMS': 10000,
        'heartbeatFrequencyMS': 10000,
        'appname': 'HeartDiseaseApp'
    }
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # Application settings
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database configuration - Force SQLite for stability
    SQLALCHEMY_DATABASE_URI = 'sqlite:///heart_disease.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Mail configuration
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True') == 'True'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')
    
    # Gemini API configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    TEMP_FOLDER = 'temp' 