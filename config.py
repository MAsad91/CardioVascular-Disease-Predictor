import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # MongoDB Atlas configuration
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