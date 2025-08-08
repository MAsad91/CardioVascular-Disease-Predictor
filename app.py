from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, make_response, session, send_file, abort, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
import os
import sys
import pandas as pd
import uuid
import base64
import re
import io
from werkzeug.utils import secure_filename
import joblib
from datetime import datetime, timedelta
# Import these modules but don't instantiate them yet
# They will be initialized in the main block
from src.heart_disease_predictor import HeartDiseasePredictor
from src.image_processor_new import MedicalReportProcessor
from src.chatbot import HeartDiseaseChatbot
from config import Config
from flask_mail import Mail, Message
from dotenv import load_dotenv
import random
from werkzeug.security import generate_password_hash
import secrets
import string

try:
    import pytesseract
    # Try to configure Tesseract path for Render deployment
    import os
    if os.path.exists('/usr/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    elif os.path.exists('/usr/local/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
except ImportError:
    pytesseract = None
    print("Warning: pytesseract not available. OCR features will be disabled.")
from fpdf import FPDF
import json
import pickle
import numpy as np
import cv2
import traceback
import matplotlib.pyplot as plt
from io import BytesIO

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.download_data import download_heart_disease_data
from src.train_model import train_and_save_model
from src.predict import predict_heart_disease, predict_quick_assessment
from src.multi_model import train_and_save_models, predict_heart_disease_multi_model
from src.explainable_ai import (
    get_feature_importance,
    get_neighbors_data,
    create_feature_importance_plot,
    create_feature_comparison_plot,
    create_patient_risk_factor_plot,
    create_key_risk_factors_chart,
    create_feature_impact_chart,
    generate_explanation,
    generate_knn_explanation,
    generate_random_forest_explanation,
    generate_xgboost_explanation,
    get_model_specific_feature_importance,
    convert_plot_to_base64
)
from src.pdf_report import generate_pdf_report

# Import models
from models import db, User, UserPrediction, UserMedicalReport, ConversationHistory

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config.from_object(Config)

# Database configuration
# Use the instance folder for database file (Flask default)
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(instance_path, exist_ok=True)
db_path = os.path.join(instance_path, 'heart_disease.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', f'sqlite:///{db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Flask-Mail configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Initialize Flask-Mail after all config
mail = Mail(app)

# Initialize database with app
db.init_app(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return User.query.get(int(user_id))

@app.context_processor
def inject_user():
    """Make current_user available in all templates"""
    return dict(current_user=current_user)

@app.context_processor
def inject_page_messages():
    """Inject page-specific messages into templates"""
    messages = []
    if 'flash_messages' in session:
        current_page = request.endpoint
        page_messages = [msg for msg in session['flash_messages'] if msg['page'] == current_page]
        messages = page_messages
        # Remove the messages that were just retrieved
        session['flash_messages'] = [msg for msg in session['flash_messages'] if msg['page'] != current_page]
    return dict(page_messages=messages)

# Keep the old Prediction model for backward compatibility with existing data
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False, unique=True)
    input_data = db.Column(db.Text, nullable=False)  # Store as JSON string
    prediction = db.Column(db.Text, nullable=False)  # Store as JSON string
    individual_predictions = db.Column(db.Text)  # Store as JSON string
    explanation = db.Column(db.Text)  # Store as JSON string
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    source = db.Column(db.String(50), default='manual_entry')  # 'manual_entry', 'pdf_upload', etc.
    risk_level = db.Column(db.String(20))  # 'Low', 'Medium', 'High'
    probability = db.Column(db.Float)  # Risk probability (0.0 to 1.0)
    
    def to_dict(self):
        """Convert the prediction to a dictionary for easy JSON serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'input_data': json.loads(self.input_data) if self.input_data else {},
            'prediction': json.loads(self.prediction) if self.prediction else {},
            'individual_predictions': json.loads(self.individual_predictions) if self.individual_predictions else {},
            'explanation': json.loads(self.explanation) if self.explanation else {},
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source': self.source,
            'risk_level': self.risk_level,
            'probability': self.probability
        }

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global variables for components (will be initialized in main)
predictor = None
report_processor = None
chatbot = None

# Decorators for role-based access control
def admin_required(f):
    """Decorator to require admin access"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if not current_user.is_admin():
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def patient_required(f):
    """Decorator to require patient access (exclude admins)"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.is_admin():
            # Store the error message in session instead of flashing it
            session['admin_access_error'] = 'This feature is only available for patients. Admins cannot access health assessment features.'
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Database helper functions
def save_prediction_to_db(session_id, input_data, consensus, individual_predictions, explanation, source='manual_entry'):
    """Save prediction data to SQLite database"""
    try:
        # Check if prediction already exists
        existing = Prediction.query.filter_by(session_id=session_id).first()
        if existing:
            print(f"Prediction with session_id {session_id} already exists, skipping database save")
            return existing
        
        # Create new prediction record
        new_prediction = Prediction(
            session_id=session_id,
            input_data=json.dumps(input_data),
            prediction=json.dumps(consensus),
            individual_predictions=json.dumps(individual_predictions),
            explanation=json.dumps(explanation),
            timestamp=datetime.utcnow(),
            source=source,
            risk_level=consensus.get('risk_level', 'Unknown'),
            probability=consensus.get('probability', 0.0)
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        print(f"Successfully saved prediction to database with session_id: {session_id}")
        return new_prediction
        
    except Exception as e:
        print(f"Error saving prediction to database: {str(e)}")
        db.session.rollback()
        return None

def get_prediction_from_db(session_id):
    """Retrieve prediction data from database"""
    try:
        prediction = Prediction.query.filter_by(session_id=session_id).first()
        return prediction.to_dict() if prediction else None
    except Exception as e:
        print(f"Error retrieving prediction from database: {str(e)}")
        return None

def get_all_predictions_from_db(limit=50):
    """Get all predictions from database with optional limit"""
    try:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()
        return [pred.to_dict() for pred in predictions]
    except Exception as e:
        print(f"Error retrieving predictions from database: {str(e)}")
        return []

def init_db():
    """Initialize the database tables"""
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            print("✅ Database tables created successfully")
            
            # Verify that tables were created
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"📋 Created tables: {tables}")
            
            # Check if users table exists
            if 'users' in tables:
                print("✅ Users table exists")
                
                # Check if any users exist
                user_count = User.query.count()
                if user_count == 0:
                    print("📝 Creating sample admin user...")
                    # Create a sample admin user
                    admin_user = User(
                        username='admin',
                        email='admin@heartcare.com',
                        first_name='Admin',
                        last_name='User',
                        role='admin',
                        is_active=True,
                        email_verified=True,
                        created_at=datetime.utcnow()
                    )
                    admin_user.set_password('admin123')
                    db.session.add(admin_user)
                    db.session.commit()
                    print("✅ Sample admin user created (username: admin, password: admin123)")
                else:
                    print(f"✅ Database has {user_count} existing user(s)")
            else:
                print("❌ Users table not found!")
                
    except Exception as e:
        print(f"❌ Error creating database tables: {str(e)}")
        import traceback
        traceback.print_exc()

# Authentication helper functions
def save_user_prediction_to_db(user_id, session_id, input_data, consensus, individual_predictions, explanation, source='manual_entry'):
    """Save prediction data for authenticated user"""
    try:
        # Check if prediction already exists
        existing = UserPrediction.query.filter_by(session_id=session_id).first()
        if existing:
            print(f"User prediction with session_id {session_id} already exists, skipping database save")
            return existing
        
        # Create new user prediction record
        new_prediction = UserPrediction(
            user_id=user_id,
            session_id=session_id,
            input_data=json.dumps(input_data),
            prediction=json.dumps(consensus),
            individual_predictions=json.dumps(individual_predictions),
            explanation=json.dumps(explanation),
            timestamp=datetime.utcnow(),
            source=source,
            risk_level=consensus.get('risk_level', 'Unknown'),
            probability=consensus.get('probability', 0.0)
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        print(f"Successfully saved user prediction to database with session_id: {session_id}")
        return new_prediction
        
    except Exception as e:
        print(f"Error saving user prediction to database: {str(e)}")
        db.session.rollback()
        return None

def get_user_predictions(user_id, limit=50):
    """Get all predictions for a specific user"""
    try:
        predictions = UserPrediction.query.filter_by(user_id=user_id).order_by(UserPrediction.timestamp.desc()).limit(limit).all()
        return [pred.to_dict() for pred in predictions]
    except Exception as e:
        print(f"Error retrieving user predictions from database: {str(e)}")
        return []

# Set up a route to serve model visualization files
@app.route('/models/<path:filename>')
def serve_model_file(filename):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    return send_from_directory(models_dir, filename)

# Feature descriptions for the UI
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male, 0 = female)',
    'cp': 'Chest pain type (0 = typical angina,\n 1 = atypical angina, 2 = non-anginal pain,\n 3 = asymptomatic)',
    'trestbps': 'Resting blood pressure in mm Hg',
    'chol': 'Serum cholesterol in mg/dl',
    'fbs': 'Fasting blood sugar > 120 mg/dl\n (1 = true, 0 = false)',
    'restecg': 'Resting electrocardiographic results\n (0 = normal, 1 = ST-T wave abnormality, 2\n = probable or definite left ventricular\n hypertrophy)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes, 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative\n to rest',
    'slope': 'Slope of the peak exercise ST segment\n (0 = upsloping, 1 = flat, 2 = downsloping)',
    'ca': 'Number of major vessels colored by\n fluoroscopy (0-3)',
    'thal': 'Thalassemia (1 = normal, 2 = fixed defect,\n 3 = reversible defect)'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Message handling function
def flash_message(message, category='info', page_specific=True):
    """Flash a message that will only show on the current page"""
    if page_specific:
        # Store message in session with current page info
        session['flash_messages'] = session.get('flash_messages', [])
        session['flash_messages'].append({
            'message': message,
            'category': category,
            'page': request.endpoint
        })
    else:
        # Use regular flash for global messages
        flash(message, category)

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    try:
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            username_or_email = request.form.get('username_or_email')
            password = request.form.get('password')
            remember = request.form.get('remember', False)
            
            if not username_or_email or not password:
                flash_message('Please fill in all fields.', 'error', page_specific=False)
                return render_template('login.html')
            
                    # Try to find user by username or email
        try:
            user = User.query.filter(
                (User.username == username_or_email) | 
                (User.email == username_or_email)
            ).first()
        except Exception as e:
            print(f"❌ Database query error in login: {str(e)}")
            flash_message('Database error. Please try again.', 'error', page_specific=False)
            return render_template('login.html')
            
            if user and user.check_password(password):
                if not user.is_active:
                    flash_message('Your account has been deactivated. Please contact support.', 'error', page_specific=False)
                    return render_template('login.html')
                
                # Update last login
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                # Log the user in
                login_user(user, remember=remember)
                
                # Store welcome message as page-specific message for dashboard
                session['flash_messages'] = session.get('flash_messages', [])
                session['flash_messages'].append({
                    'message': f'Welcome back, {user.get_full_name()}!',
                    'category': 'success',
                    'page': 'index'  # Explicitly set for dashboard page
                })
                
                # Redirect to next page or index
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('index'))
            else:
                flash_message('Invalid username/email or password.', 'error', page_specific=False)
        
        # Get logout message from session if exists
        logout_message = session.pop('logout_message', None)
        return render_template('login.html', logout_message=logout_message)
        
    except Exception as e:
        print(f"❌ Login route error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash_message('An error occurred. Please try again.', 'error', page_specific=False)
        return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('user_name')  # updated to match form field
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all required fields.', 'error')
            return render_template('signup.html')
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash('Invalid email address.', 'error')
            return render_template('signup.html')
        
        # Validate username format
        username_pattern = r'^[a-zA-Z0-9_]+$'
        if not re.match(username_pattern, username):
            flash('Username can only contain letters, numbers, and underscores.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')
        
        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists.', 'error')
            return render_template('signup.html')
        
        # Create new user (no role field)
        new_user = User(
            username=username,
            email=email,
            created_at=datetime.utcnow(),
            is_active=True,
            email_verified=False
        )
        new_user.set_password(password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"Error creating user: {e}")
            flash('An error occurred while creating your account. Please try again.', 'error')
            return render_template('signup.html')
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    # Store logout message in session instead of flash
    session['logout_message'] = 'You have been logged out successfully.'
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter your email address.', 'error')
            return render_template('forgot_password.html')
        try:
            user = User.query.filter_by(email=email).first()
        except Exception as e:
            print(f"❌ Database query error in forgot_password: {str(e)}")
            flash('Database error. Please try again.', 'error')
            return render_template('forgot_password.html')
        if not user:
            flash('No user found with that email address.', 'error')
            return render_template('forgot_password.html')
        # Generate 6-character password with at least one letter and one digit
        letters = string.ascii_letters
        digits = string.digits
        alphabet = letters + digits
        password_chars = [secrets.choice(letters), secrets.choice(digits)]
        password_chars += [secrets.choice(alphabet) for _ in range(4)]
        secrets.SystemRandom().shuffle(password_chars)
        new_password = ''.join(password_chars)
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        # Send email
        try:
            msg = Message('Heart Care Password Reset', recipients=[user.email])
            msg.body = f'Your new password is: {new_password}\nPlease log in and change it immediately.'  # Fallback plain text
            msg.html = f"""
            <div style=\"font-family: Arial, sans-serif; background: #f7fafc; padding: 32px;\">
              <div style=\"max-width: 480px; margin: auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px #e2e8f0; padding: 32px;\">
                <div style=\"text-align: center;\">
                  <div style=\"font-size: 48px; margin-bottom: 8px;\">🔒❤️</div>
                  <h2 style=\"color: #2b6cb0; margin-bottom: 0.5em;\">Heart Care Password Reset</h2>
                </div>
                <p style=\"font-size: 1.1em; color: #2d3748; text-align: center;\">
                  Your password has been reset. Please use the new password below to log in:
                </p>
                <div style=\"background: #f0fff4; border: 1px solid #38a169; border-radius: 8px; padding: 18px; margin: 24px 0; text-align: center;\">
                  <span style=\"font-size: 1.3em; letter-spacing: 2px; color: #22543d;\">
                    <b>🔑 {new_password}</b>
                  </span>
                </div>
                <p style=\"color: #4a5568; text-align: center;\">
                  <b>Tip:</b> For your security, please log in and change your password immediately after signing in.
                </p>
                <div style=\"text-align: center; margin-top: 32px; color: #718096;\">
                  <span style=\"font-size: 1.2em;\">💙 Stay healthy,<br>Heart Care Team</span>
                </div>
              </div>
            </div>
            """
            print(f"[DEBUG] mail object: {mail}")
            mail.send(msg)
            flash('A new password has been sent to your email address. Please check your email and log in with the new password.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            import traceback
            print('[ERROR] Exception in forgot_password email sending:')
            traceback.print_exc()
            flash(f'Error sending email: {str(e)}', 'error')
        return render_template('forgot_password.html')
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token"""
    # Verify token
    if 'reset_token' not in session or session['reset_token'] != token:
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('forgot_password'))
    
    user_id = session.get('reset_user_id')
    if not user_id:
        flash('Invalid reset session.', 'error')
        return redirect(url_for('forgot_password'))
    
    user = User.query.get(user_id)
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not new_password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('reset_password.html', token=token)
        
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', token=token)
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('reset_password.html', token=token)
        
        # Update password
        try:
            user.set_password(new_password)
            db.session.commit()
            
            # Clear session
            session.pop('reset_user_id', None)
            session.pop('reset_token', None)
            
            flash('Your password has been reset successfully! Please log in with your new password.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while resetting your password. Please try again.', 'error')
    
    return render_template('reset_password.html', token=token, user=user)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Allow logged-in users to change their password"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([current_password, new_password, confirm_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('change_password.html')
        
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'error')
            return render_template('change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('change_password.html')
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('change_password.html')
        
        if new_password == current_password:
            flash('New password must be different from current password.', 'error')
            return render_template('change_password.html')
        
        try:
            current_user.set_password(new_password)
            db.session.commit()
            flash('Password changed successfully!', 'success')
            return redirect(url_for('profile'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while changing your password. Please try again.', 'error')
    
    return render_template('change_password.html')

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    user_predictions = get_user_predictions(current_user.id, limit=10)
    return render_template('profile.html', user=current_user, predictions=user_predictions)




# Global flag to track initialization status
_initialized = False

def initialize_app():
    """Initialize the application components"""
    global _initialized, predictor, report_processor, chatbot
    
    if _initialized:
        return True
    
    try:
        print("🔧 Initializing application components...")
        
        # Create necessary directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('temp', exist_ok=True)
        
        # Initialize database
        with app.app_context():
            init_db()
            print("✅ Database initialized")
        
        # Initialize components (skip model training for now to avoid blocking)
        try:
            predictor = HeartDiseasePredictor()
            print("✅ Heart Disease Predictor initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize Heart Disease Predictor: {str(e)}")
            predictor = None
        
        try:
            report_processor = MedicalReportProcessor()
            print("✅ Medical Report Processor initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize Medical Report Processor: {str(e)}")
            report_processor = None
        
        try:
            chatbot = HeartDiseaseChatbot()
            print("✅ Heart Disease Chatbot initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize Heart Disease Chatbot: {str(e)}")
            chatbot = None
        
        _initialized = True
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize app: {str(e)}")
        return False

@app.route('/')
@login_required
def index():
    """Main dashboard with dynamic system statistics."""
    # Initialize app if not already done
    if not _initialized:
        initialize_app()
    
    print("[DEBUG] ===== INDEX ROUTE CALLED =====")
    print(f"[DEBUG] Current user: {current_user}")
    print(f"[DEBUG] User authenticated: {current_user.is_authenticated}")
    
    try:
        print("[DEBUG] Index route called - starting dashboard load")
        
        # Model metrics
        metrics = {}
        metrics_path = os.path.join('models', 'metrics.pkl')
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
        else:
            metrics = {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1': 0.85
            }

        # Total assessments (all predictions) - ADD DEBUG PRINTS HERE
        print("[DEBUG] About to query database...")
        try:
            prediction_count = Prediction.query.count()
        except Exception as e:
            print(f"Error counting old predictions: {str(e)}")
            prediction_count = 0
            
        user_prediction_count = UserPrediction.query.count()
        total_assessments = prediction_count + user_prediction_count
        
        print(f"[DEBUG] Index - Prediction table count: {prediction_count}")
        print(f"[DEBUG] Index - UserPrediction table count: {user_prediction_count}")
        print(f"[DEBUG] Index - Total assessments: {total_assessments}")

        # Model accuracy (from metrics)
        model_accuracy = metrics.get('accuracy', 0.0)
        print(f"[DEBUG] Index - Model accuracy: {model_accuracy}")

        # Reports analyzed (all uploaded reports)
        try:
            reports_analyzed = UserMedicalReport.query.count()
            print(f"[DEBUG] Index - Reports analyzed: {reports_analyzed}")
        except Exception as e:
            print(f"Error counting medical reports: {str(e)}")
            reports_analyzed = 0

        # Active users (show total users as active)
        try:
            active_users = User.query.count()
            print(f"[DEBUG] Index - Total users (active): {active_users}")
        except Exception as e:
            print(f"Error counting active users: {str(e)}")
            active_users = 0

        print(f"[DEBUG] Index - Active users: {active_users}")
        print(f"[DEBUG] Index - Dashboard data prepared successfully")
        print(f"[DEBUG] Index - About to render template with total_assessments: {total_assessments}")
        
        # Get admin access error message from session if exists
        admin_access_error = session.pop('admin_access_error', None)
        
        return render_template('index.html',
                             metrics=metrics,
                             total_assessments=total_assessments,
                             model_accuracy=model_accuracy,
                             reports_analyzed=reports_analyzed,
                             active_users=active_users,
                             admin_access_error=admin_access_error)
    except Exception as e:
        print(f"[ERROR] Index route error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('An error occurred while loading the dashboard.', 'error')
        return redirect(url_for('index'))



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/test_messages')
def test_messages():
    """Test route to verify message handling"""
    # Test different types of messages
    flash_message('This is a page-specific error message', 'error', page_specific=True)
    flash_message('This is a page-specific success message', 'success', page_specific=True)
    flash_message('This is a global info message', 'info', page_specific=False)
    return render_template('test_messages.html')

@app.route('/test_modal')
def test_modal():
    """Test route to verify modal system"""
    flash('This is a test flash message', 'error')
    flash('This is another test message', 'success')
    return render_template('test_modal.html')

@app.route('/explain_prediction/<session_id>', methods=['GET'])
@login_required
@patient_required
def explain_prediction(session_id):
    try:
        temp_file = f'temp/prediction_{session_id}.pkl'
        print(f"DEBUG: Loading prediction file: {temp_file}")
        if not os.path.exists(temp_file):
            print("DEBUG: Prediction file does not exist.")
            return render_template('error.html', error="Prediction data not found. Please make a new prediction.")
        temp_data = pd.read_pickle(temp_file)
        print("DEBUG: Loaded temp_data:", temp_data)
        input_data = temp_data.get('input_data')
        prediction = temp_data.get('prediction')
        individual_predictions = temp_data.get('individual_predictions', {})
        explanation = temp_data.get('explanation', {})
        assessment_type = explanation.get('assessment_type')

        # For quick assessment, skip all detailed model logic and only pass minimal variables
        if assessment_type in ['quick_form', 'quick_assessment']:
            return render_template('explanation.html', explanation=explanation, session_id=session_id)

        print("DEBUG: input_data:", input_data)
        print("DEBUG: prediction:", prediction)
        print("DEBUG: individual_predictions:", individual_predictions)
        print("DEBUG: explanation:", explanation)
        if not isinstance(input_data, dict) or not isinstance(prediction, dict):
            print("DEBUG: input_data or prediction is not a dict.")
            return render_template('error.html', error="Invalid prediction data. Please make a new prediction.")
        
        # Ensure explanation is a dictionary and has 'model_insights' (and other critical keys) initialized
        if not isinstance(explanation, dict):
            print("DEBUG: Explanation loaded is not a dictionary. Re-initializing.")
            explanation = {}
            explanation['visualization_error'] = "Explanation data was corrupted. Some visualizations may be missing."
        
        if 'model_insights' not in explanation or not isinstance(explanation['model_insights'], dict):
            explanation['model_insights'] = {}
        
        # Ensure all values are properly defined before processing
        if not prediction or not input_data:
            return render_template('error.html', error="Invalid prediction data. Please make a new prediction.")
        
        # Ensure probability is a valid number
        if 'probability' not in prediction or not isinstance(prediction['probability'], (int, float)):
            prediction['probability'] = 0.0
        
        # Get consensus prediction (prioritize from explanation if available, fallback to prediction)
        consensus = explanation.get('consensus') or prediction.get('consensus') or prediction.get('risk_level', 'Unknown')
        
        # Attempt to generate and add visualizations to the existing explanation
        try:
            # Get feature importance from all models
            feature_importance = get_feature_importance('models')
            neighbors_data = get_neighbors_data(input_data)
            
            # Create visualizations - ALWAYS regenerate to ensure they exist
            try:
                print(f"DEBUG: Creating feature_importance_plot...")
                print(f"DEBUG: feature_importance type: {type(feature_importance)}")
                print(f"DEBUG: feature_importance shape: {feature_importance.shape if hasattr(feature_importance, 'shape') else 'No shape'}")
                print(f"DEBUG: feature_importance columns: {feature_importance.columns if hasattr(feature_importance, 'columns') else 'No columns'}")
                
                feature_importance_plot = create_feature_importance_plot(feature_importance)
                print(f"DEBUG: feature_importance_plot created: {feature_importance_plot is not None}")
                
                if feature_importance_plot is not None:
                    explanation['feature_importance_img'] = convert_plot_to_base64(feature_importance_plot)
                    print(f"DEBUG: feature_importance_img created successfully, length: {len(explanation['feature_importance_img'])}")
                else:
                    print("DEBUG: feature_importance_plot returned None - creating fallback")
                    # Create a simple fallback plot
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # Create a simple fallback plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    features = ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 'Heart Rate']
                    importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
                    
                    y_pos = range(len(features))
                    bars = ax.barh(y_pos, importance, color='skyblue', alpha=0.8, height=0.6)
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(features, fontsize=10)
                    ax.set_xlabel('Importance Score', fontsize=12)
                    ax.set_title('Feature Importance (Fallback)', fontsize=14, fontweight='bold')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    
                    explanation['feature_importance_img'] = convert_plot_to_base64(fig)
                    plt.close(fig)
                    print(f"DEBUG: Fallback feature_importance_img created, length: {len(explanation['feature_importance_img'])}")
                    
            except Exception as e:
                print(f"DEBUG: Error creating feature_importance_plot: {str(e)}")
                import traceback
                traceback.print_exc()
                explanation['feature_importance_img'] = None
            
            # ALWAYS regenerate all visualizations to ensure they exist
            feature_comparison_plot = create_feature_comparison_plot(neighbors_data, input_data)
            explanation['feature_comparison_img'] = convert_plot_to_base64(feature_comparison_plot)
            
            # ALWAYS regenerate risk_factor_img to ensure we use the enhanced version
            risk_factor_plot = create_feature_impact_chart(input_data, feature_importance)
            explanation['risk_factor_img'] = convert_plot_to_base64(risk_factor_plot)
            
            # ALWAYS regenerate key_risk_factors_img
            key_risk_factors_chart = create_key_risk_factors_chart(input_data, feature_importance)
            explanation['key_risk_factors_img'] = convert_plot_to_base64(key_risk_factors_chart)
            
            # Debugging: Check if image data is present
            print(f"Debug: feature_importance_img length: {len(explanation.get('feature_importance_img', '')) if explanation.get('feature_importance_img') else 0}")
            print(f"Debug: feature_comparison_img length: {len(explanation.get('feature_comparison_img', '')) if explanation.get('feature_comparison_img') else 0}")
            print(f"Debug: risk_factor_img length: {len(explanation.get('risk_factor_img', '')) if explanation.get('risk_factor_img') else 0}")
            print(f"Debug: key_risk_factors_img length: {len(explanation.get('key_risk_factors_img', '')) if explanation.get('key_risk_factors_img') else 0}")
            
            # Add model-specific feature importance plots
            rf_feature_importance = get_model_specific_feature_importance('models', 'random_forest')
            xgb_feature_importance = get_model_specific_feature_importance('models', 'xgboost')
            
            rf_importance_plot = create_feature_importance_plot(rf_feature_importance)
            xgb_importance_plot = create_feature_importance_plot(xgb_feature_importance)
            
            explanation['rf_feature_importance'] = convert_plot_to_base64(rf_importance_plot)
            explanation['xgb_feature_importance'] = convert_plot_to_base64(xgb_importance_plot)
            
            # Get neighbors data for KNN explanation
            neighbors_data = get_neighbors_data(input_data) if neighbors_data is None else neighbors_data
            
            # Ensure boxplots exist in explanation
            if 'boxplots' not in explanation:
                print("WARNING: Boxplots missing from explanation data. They should have been generated during prediction.")
            
            explanation['model_insights'] = {
                'knn': generate_knn_explanation(input_data, consensus, neighbors_data, None),
                'random_forest': generate_random_forest_explanation(input_data, consensus, None),
                'xgboost': generate_xgboost_explanation(input_data, consensus, None)
            }
            
        except Exception as e:
            print(f"Error generating visualizations in explain_prediction: {str(e)}")
            # DO NOT set visualization fields to None if they already exist
            if not explanation.get('feature_importance_img'):
                explanation['feature_importance_img'] = None
            if not explanation.get('feature_comparison_img'):
                explanation['feature_comparison_img'] = None
            if not explanation.get('risk_factor_img'):
                explanation['risk_factor_img'] = None
            if not explanation.get('key_risk_factors_img'):
                explanation['key_risk_factors_img'] = None
            if not explanation.get('rf_feature_importance'):
                explanation['rf_feature_importance'] = None
            if not explanation.get('xgb_feature_importance'):
                explanation['xgb_feature_importance'] = None
            explanation['visualization_error'] = "Unable to generate some visualizations. Please try again later."
        
        # Add the session_id to the explanation for the PDF download link
        explanation['id'] = session_id
        
        # Ensure all required fields are present (these should already be in the loaded explanation or defaults)
        explanation['risk_level'] = explanation.get('risk_level', prediction.get('risk_level', 'Unknown'))
        explanation['risk_type'] = explanation.get('risk_type', prediction.get('risk_type', 'Unknown'))
        explanation['probability'] = explanation.get('probability', prediction.get('probability', 0.0))
        explanation['risk_description'] = explanation.get('risk_description', prediction.get('risk_description', 'No description available.'))
        
        # Extract values for template variables
        risk_level = explanation.get('risk_level', 'Unknown')
        probability = explanation.get('probability', 0.0)
        
        # Extract individual model probabilities for the template
        knn_probability = individual_predictions.get('knn', {}).get('probability', 0.0) * 100
        rf_probability = individual_predictions.get('random_forest', {}).get('probability', 0.0) * 100
        xgb_probability = individual_predictions.get('xgboost', {}).get('probability', 0.0) * 100
        
        # Debug: Check if boxplots were regenerated
        print(f"DEBUG: Final explanation boxplots keys: {list(explanation.get('boxplots', {}).keys())}")
        print(f"DEBUG: Boxplot version: {explanation.get('boxplot_version', 'None')}")
        print(f"DEBUG: Number of boxplots: {len(explanation.get('boxplots', {}))}")
        print(f"DEBUG: Explanation keys: {list(explanation.keys())}")
        print(f"DEBUG: Boxplots exist: {'boxplots' in explanation}")
        if 'boxplots' in explanation:
            print(f"DEBUG: Boxplots is dict: {isinstance(explanation['boxplots'], dict)}")
            print(f"DEBUG: Boxplots is empty: {len(explanation['boxplots']) == 0}")
        
        print(f"DEBUG: Session ID: {session_id}")
        print(f"DEBUG: Template being rendered: explanation.html")
        print(f"DEBUG: Cache buster: {datetime.now().timestamp()}")
        
        # Debugging: Check what's being passed to template
        print(f"DEBUG: Final explanation keys: {list(explanation.keys())}")
        print(f"DEBUG: feature_importance_img exists: {'feature_importance_img' in explanation}")
        print(f"DEBUG: feature_importance_img is None: {explanation.get('feature_importance_img') is None}")
        print(f"DEBUG: feature_importance_img type: {type(explanation.get('feature_importance_img'))}")
        if explanation.get('feature_importance_img'):
            print(f"DEBUG: feature_importance_img length: {len(explanation['feature_importance_img'])}")
            print(f"DEBUG: feature_importance_img starts with: {explanation['feature_importance_img'][:50]}...")
        
        # Render the merged results and explanation page
        return render_template('explanation.html',
            explanation=explanation,
            session_id=session_id,
            knn_probability=knn_probability,
            rf_probability=rf_probability,
            xgb_probability=xgb_probability,
            knn_risk=individual_predictions.get('knn', {}).get('risk_level', 'Unknown'),
            rf_risk=individual_predictions.get('random_forest', {}).get('risk_level', 'Unknown'),
            xgb_risk=individual_predictions.get('xgboost', {}).get('risk_level', 'Unknown'),
            cache_buster=datetime.now().timestamp(),  # Force cache refresh
        )
    
    except Exception as e:
        print(f"Error in explain_prediction: {str(e)}")
        return render_template('error.html', error="An error occurred while generating the explanation.")

def save_prediction_result(result_id, data):
    """Save prediction result to a temporary file."""
    temp_file = os.path.join(app.config['TEMP_FOLDER'], f'prediction_{result_id}.pkl')
    with open(temp_file, 'wb') as f:
        pickle.dump(data, f)
    return temp_file

@app.route('/upload', methods=['GET', 'POST'])
@login_required
@patient_required
def upload_report():
    try:
        if request.method == 'GET':
            return render_template('upload.html')

        # Handle file uploads
        uploaded_files = request.files.getlist('file')  # Get list of files
        if not uploaded_files or all(not file.filename for file in uploaded_files):
            flash('No files uploaded')
            return render_template('upload.html', warning='Please select at least one file to upload')

        # Process each uploaded file
        all_extracted_data = []
        warnings = []
        form_data = {}  # Initialize form_data at the top level
        processor = MedicalReportProcessor()
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']  # Ensure 'fbs' is included
        missing_fields = []

        for uploaded_file in uploaded_files:
            if uploaded_file and uploaded_file.filename:
                try:
                    # Create a secure filename
                    original_filename = secure_filename(uploaded_file.filename)
                    unique_filename = f"{str(uuid.uuid4())}_{original_filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    
                    try:
                        # Save the file
                        uploaded_file.save(file_path)
                        print(f"File saved to: {file_path}")  # Debug log
                        
                        # Process the file
                        extracted_data = processor.process_report(file_path)
                        print(f"\nProcessed data: {extracted_data}")  # Debug log
                        
                        if extracted_data:
                            # Copy fields directly from extracted_data (case-insensitive match)
                            for field in required_fields:
                                for k in extracted_data:
                                    if k.lower() == field.lower():
                                        value = extracted_data[k]
                                        # Special handling for sex
                                        if field == 'sex':
                                            if isinstance(value, str):
                                                if value.lower() == 'male':
                                                    value = 1
                                                elif value.lower() == 'female':
                                                    value = 0
                                        # Special handling for trestbps (extract systolic if in '120/80' format)
                                        if field == 'trestbps' and isinstance(value, str) and '/' in value:
                                            try:
                                                value = float(value.split('/')[0])
                                            except Exception:
                                                value = None
                                        # Ensure correct type for chol, thalach
                                        if field in ['chol', 'thalach'] and isinstance(value, str):
                                            try:
                                                value = float(value)
                                            except Exception:
                                                value = None
                                        # Special handling for fbs (fasting blood sugar)
                                        if field == 'fbs':
                                            # If value is numeric, map to dropdown value
                                            try:
                                                fbs_num = float(value)
                                                if fbs_num > 120:
                                                    value = '1'
                                                else:
                                                    value = '0'
                                            except Exception:
                                                # If value is already '0' or '1', keep as is
                                                if str(value).strip() in ['0', '1']:
                                                    value = str(value).strip()
                                                # If value is a label, map to correct value
                                                elif isinstance(value, str):
                                                    if 'normal' in value.lower() or '≤' in value or '<=' in value:
                                                        value = '0'
                                                    elif 'high' in value.lower() or '>' in value:
                                                        value = '1'
                                        form_data[field] = value
                                        print(f"Copied {field}: {value}")  # Debug log
                                        break
                            print(f"DEBUG: form_data after copying: {form_data}")  # Debug log
                            # Determine missing fields for this file
                            missing = [f for f in required_fields if f not in form_data or form_data[f] in [None, '', 'None']]
                            missing_fields.extend(missing)
                            # Append to all_extracted_data for success message
                            all_extracted_data.append(extracted_data)
                            # Save medical report to database
                            try:
                                img_for_type_detection = cv2.imread(file_path) # Reload image for type detection
                                if img_for_type_detection is not None:
                                    text_for_type_detection = processor.extract_text(processor.preprocess_image(img_for_type_detection))
                                    detected_report_type = processor.detect_report_type(text_for_type_detection)
                                else:
                                    detected_report_type = 'unknown'

                                new_report = UserMedicalReport(
                                    user_id=current_user.id,
                                    filename=unique_filename,
                                    original_filename=original_filename,
                                    file_type='pdf',
                                    upload_date=datetime.utcnow(),
                                    content=None,  # You can set this to extracted text if available
                                    analysis_results=json.dumps(extracted_data),
                                    is_processed=True,
                                    file_size=os.path.getsize(file_path)
                                )
                                db.session.add(new_report)
                                db.session.commit()
                                print(f"Successfully saved medical report for user {current_user.id}: {original_filename}")
                            except Exception as db_e:
                                print(f"Error saving medical report to database: {db_e}")
                                traceback.print_exc()
                                warnings.append(f"Error saving {original_filename} to database.")
                
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        warnings.append(f"Error processing {original_filename}: {str(e)}")
                        
                except Exception as e:
                    print(f"Error handling file {uploaded_file.filename}: {str(e)}")
                    warnings.append(f"Error handling {uploaded_file.filename}: {str(e)}")

        # Store form_data in session, preserving numeric types
        print(f"\nDEBUG: Form data before session storage: {form_data}")  # Debug log
        session['prefilled_data'] = {}
        for k, v in form_data.items():
            if v is not None:  # Only store non-None values
                if isinstance(v, (int, float)):
                    session['prefilled_data'][k] = v  # Store numbers as is
                else:
                    session['prefilled_data'][k] = str(v)  # Convert other types to string
        print(f"DEBUG: Session data after storage: {session['prefilled_data']}")  # Debug log

        # Full list of fields required for assessment
        assessment_required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        # Calculate missing_fields as all assessment-required fields not present in form_data
        missing_fields = [f for f in assessment_required_fields if f not in form_data or form_data[f] in [None, '', 'None']]

        # Store warnings, missing_fields, and extracted_fields in session for use in risk_assessment
        session['upload_warnings'] = warnings
        session['upload_missing_fields'] = missing_fields
        session['upload_extracted_fields'] = list(form_data.keys())
        session['upload_prefilled_data'] = form_data

        # After copying all fields, ensure 'ca' is a string integer for dropdown compatibility
        if 'ca' in form_data and form_data['ca'] is not None:
            try:
                form_data['ca'] = str(int(float(form_data['ca'])))
            except Exception:
                form_data['ca'] = None

        # Always redirect to risk_assessment after upload
        return redirect(url_for('risk_assessment'))

    except Exception as e:
        print(f"Error in upload_report: {str(e)}")
        traceback.print_exc()
        flash(f'An unexpected error occurred: {str(e)}', 'error')
        return render_template('upload.html')

@app.route('/train', methods=['GET'])
def train():
    try:
        # Check if data exists, if not download it
        data_path = os.path.join('data', 'heart_disease.csv')
        if not os.path.exists(data_path):
            download_heart_disease_data()
        
        # Train the models
        models, metrics = train_and_save_models(data_path, 'models')
        
        # Load metrics for display
        all_metrics_path = os.path.join('models', 'all_metrics.pkl')
        if os.path.exists(all_metrics_path):
            all_metrics = joblib.load(all_metrics_path)
            metrics = all_metrics.get('knn', metrics)  # Fallback to basic metrics if needed
        
        return render_template('train.html', metrics=metrics)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get input data from JSON request
        input_data = request.json
        
        # Make prediction using multiple models
        consensus, individual_predictions = predict_heart_disease_multi_model(input_data, 'models')
        
        # Return consensus and individual predictions
        return jsonify({
            'consensus': consensus,
            'individual_predictions': individual_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/reports')
@login_required
@patient_required
def reports():
    """View all stored predictions from database for the current user"""
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # Get predictions for the current user
        all_predictions = get_user_predictions(current_user.id, limit=per_page * 10)
        print(f"[DEBUG] User {current_user.id} has {len(all_predictions)} predictions:")
        for pred in all_predictions:
            print(f"  Session ID: {pred.get('session_id')}, Timestamp: {pred.get('timestamp')}, Source: {pred.get('source')}, User ID: {pred.get('user_id')}")
        
        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        predictions = all_predictions[start_idx:end_idx]
        
        # Format for display
        formatted_predictions = []
        for pred in predictions:
            try:
                formatted_prediction = {
                    'id': pred['session_id'],
                    'date': pred['timestamp'][:19] if pred['timestamp'] else 'Unknown',
                    'risk_level': pred['risk_level'] or 'Unknown',
                    'probability': int((pred['probability'] or 0) * 100),
                    'source': pred['source'],
                    'metrics': {
                        'cholesterol': pred['input_data'].get('chol', 0),
                        'blood_pressure': pred['input_data'].get('trestbps', 0),
                        'heart_rate': pred['input_data'].get('thalach', 0),
                        'st_depression': pred['input_data'].get('oldpeak', 0)
                    }
                }
                formatted_predictions.append(formatted_prediction)
            except Exception as e:
                print(f"Error formatting prediction: {str(e)}")
                continue
        
        return render_template('reports.html', 
                             predictions=formatted_predictions,
                             page=page,
                             per_page=per_page,
                             total_predictions=len(all_predictions))
    
    except Exception as e:
        print(f"Error in reports route: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/api/predictions', methods=['GET'])
def api_get_predictions():
    """API endpoint to get predictions from database"""
    try:
        limit = int(request.args.get('limit', 50))
        predictions = get_all_predictions_from_db(limit=limit)
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predictions/<session_id>', methods=['GET'])
def api_get_prediction(session_id):
    """API endpoint to get a specific prediction by session_id"""
    try:
        prediction = get_prediction_from_db(session_id)
        if prediction:
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reports/search', methods=['GET'])
def search_reports():
    """Search reports (currently using file system, can be enhanced with database)"""
    try:
        # Get search parameters
        risk_level = request.args.get('risk_level')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        limit = int(request.args.get('limit', 50))
        
        # For now, get all predictions and filter (can be optimized with SQL queries later)
        all_predictions = get_all_predictions_from_db(limit=limit * 2)
        filtered_predictions = []
        
        for pred in all_predictions:
            # Apply filters
            if risk_level and pred['risk_level'] != risk_level:
                continue
            if date_from and pred['timestamp'] < date_from:
                continue
            if date_to and pred['timestamp'] > date_to:
                continue
            filtered_predictions.append(pred)
        
        return jsonify({
            'success': True,
            'predictions': filtered_predictions[:limit],
            'count': len(filtered_predictions)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reports/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    """Delete a prediction from database"""
    try:
        prediction = Prediction.query.filter_by(session_id=report_id).first()
        if prediction:
            db.session.delete(prediction)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Prediction deleted successfully'})
        return jsonify({'success': False, 'error': 'Prediction not found'}), 404
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/comparative_analysis')
@login_required
@patient_required
def comparative_analysis():
    try:
        print("Starting comparative analysis...")  # Debug log
        # Get all predictions for the current user from the database
        predictions = get_user_predictions(current_user.id)
        print(f"Loaded {len(predictions)} user predictions from DB")  # Debug log

        # Build metrics dict from actual user data for each prediction
        for pred in predictions:
            input_data = pred.get('input_data', {})
            prediction = pred.get('prediction', {})
            pred['metrics'] = {
                'cholesterol': input_data.get('chol', 0),
                'blood_pressure': input_data.get('trestbps', 0),
                'heart_rate': input_data.get('thalach', 0),
                'st_depression': input_data.get('oldpeak', 0),
                'risk_score': prediction.get('probability', 0) * 100
            }
        
        # Patch: Ensure every prediction has a metrics dict with all expected keys
        expected_metrics = ['cholesterol', 'blood_pressure', 'heart_rate', 'st_depression', 'risk_score']
        for pred in predictions:
            if 'metrics' not in pred or not isinstance(pred['metrics'], dict):
                pred['metrics'] = {}
            for key in expected_metrics:
                if key not in pred['metrics']:
                    pred['metrics'][key] = 0

        # Prepare data for visualization
        timeline_data = prepare_timeline_data(predictions)
        print(f"Timeline data: {timeline_data}")  # Debug log

        # Check if we have enough data for visualization
        if not predictions or len(predictions) < 2:
            print("Not enough predictions for visualization")  # Debug log
            return render_template('comparative_analysis.html', 
                                  predictions=[], 
                                  timeline_data=timeline_data,
                                  feature_descriptions=feature_descriptions)

        return render_template('comparative_analysis.html', 
                              predictions=predictions, 
                              timeline_data=timeline_data,
                              feature_descriptions=feature_descriptions)
    
    except Exception as e:
        print(f"Comparative analysis error: {str(e)}")
        return render_template('error.html', error=str(e))

def prepare_timeline_data(predictions):
    """Prepare data for timeline visualization"""
    print(f"Preparing timeline data from {len(predictions)} predictions")  # Debug log
    
    if not predictions:
        empty_data = {
            'dates': [],
            'cholesterol': [],
            'blood_pressure': [],
            'heart_rate': [],
            'st_depression': [],
            'risk_score': []
        }
        print("No predictions found, returning empty data")  # Debug log
        return empty_data
    
    # Sort predictions by date (oldest first)
    chronological = sorted(predictions, key=lambda x: x.get('timestamp', ''))
    print(f"Sorted predictions chronologically: {chronological}")  # Debug log
    
    timeline_data = {
        'dates': [],
        'cholesterol': [],
        'blood_pressure': [],
        'heart_rate': [],
        'st_depression': [],
        'risk_score': []
    }
    
    for pred in chronological:
        # Extract date for x-axis
        if isinstance(pred.get('timestamp'), str):
            date = pred.get('timestamp').split(' ')[0]
        else:
            date = pred.get('date', 'Unknown')
        
        timeline_data['dates'].append(date)
        
        # Extract metrics
        metrics = pred.get('metrics', {})
        print(f"Processing metrics for date {date}: {metrics}")  # Debug log
        
        timeline_data['cholesterol'].append(metrics.get('cholesterol', 0))
        timeline_data['blood_pressure'].append(metrics.get('blood_pressure', 0))
        timeline_data['heart_rate'].append(metrics.get('heart_rate', 0))
        timeline_data['st_depression'].append(metrics.get('st_depression', 0))
        timeline_data['risk_score'].append(metrics.get('risk_score', 0))
    
    print(f"Final timeline data: {timeline_data}")  # Debug log
    return timeline_data

@app.route('/chat', methods=['GET', 'POST'])
@login_required
@patient_required
def chat():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        try:
            # Check if chatbot is initialized
            if chatbot is None:
                return jsonify({
                    'error': 'Chatbot is not available. Please try again later.'
                }), 503
            
            # Process the user's response
            next_question, health_data = chatbot.process_response(user_input)
            
            if health_data is not None:
                # All questions answered, make prediction
                try:
                    # Convert health data to the format expected by the predictor
                    input_data = {
                        'age': float(health_data['age']),
                        'sex': int(health_data['sex']),
                        'trestbps': float(health_data['trestbps']),
                        'chol': float(health_data['chol']),
                        'thalach': float(health_data['thalach']),
                        'oldpeak': float(health_data['oldpeak'])
                    }
                    
                    # Make prediction using multiple models
                    consensus, individual_predictions = predict_heart_disease_multi_model(input_data, 'models')
                    
                    # Determine risk level based on probability
                    probability = max(0.0, min(1.0, consensus['probability']))
                    if probability >= 0.7:
                        risk_level = 'High'
                        risk_description = "High risk of heart disease. Immediate medical consultation is recommended."
                    elif probability >= 0.4:
                        risk_level = 'Medium'
                        risk_description = "Medium risk of heart disease. Regular check-ups advised."
                    else:
                        risk_level = 'Low'
                        risk_description = "Low risk of heart disease. Maintain a healthy lifestyle."
                    
                    return jsonify({
                        'message': f"Thank you for completing the assessment.\n\nRisk Level: {risk_level}\n{risk_description}\n\nFor a more detailed assessment, please use the form assessment.",
                        'completed': True
                    })
                except Exception as e:
                    print(f"Error making prediction: {str(e)}")
                    return jsonify({
                        'message': "Thank you for completing the assessment. For a more detailed evaluation, please use the form assessment.",
                        'completed': True
                    })
            
            if next_question is None:
                return jsonify({'error': 'Invalid response'}), 400
            
            return jsonify({'message': next_question})
            
        except Exception as e:
            print(f"Error in chat route: {str(e)}")  # Log the error
            return jsonify({
                'error': f"An error occurred while processing your response. Please try again."
            }), 500
    
    # GET request - start new chat
    if chatbot is None:
        flash('Chatbot is not available. Please try again later.', 'error')
        return redirect(url_for('index'))
    
    chatbot.reset()
    initial_question = chatbot.get_current_question()['message']
    return render_template('chat.html', initial_question=initial_question)

@app.route('/test_timeline')
def test_timeline():
    try:
        # Create test prediction data
        test_data = {
            'input_data': {
                'chol': 200,
                'trestbps': 120,
                'thalach': 80,
                'oldpeak': 1.0
            },
            'prediction': {
                'risk_level': 'Low',
                'probability': 0.2
            },
            'timestamp': '2024-03-15 10:00:00'
        }
        
        # Save test data
        if not os.path.exists('temp'):
            os.makedirs('temp')
        pd.to_pickle(test_data, 'temp/prediction_test.pkl')
        
        # Call comparative analysis
        return comparative_analysis()
    except Exception as e:
        print(f"Test timeline error: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/health_tips', methods=['GET'])
def health_tips():
    age = request.args.get('age', type=int)
    tips = []
    if age is not None:
        if age < 20:
            tips = [
                "Stay active with daily play and exercise.",
                "Eat fruits, vegetables, and whole grains.",
                "Avoid sugary drinks and junk food.",
                "Get enough sleep every night."
            ]
        elif age < 40:
            tips = [
                "Exercise regularly (at least 150 minutes per week).",
                "Maintain a healthy weight.",
                "Don't smoke and avoid secondhand smoke.",
                "Manage stress with healthy activities."
            ]
        elif age < 60:
            tips = [
                "Monitor your blood pressure and cholesterol.",
                "Eat a balanced diet low in salt and saturated fat.",
                "Stay physically active and manage your weight.",
                "Get regular health check-ups."
            ]
        else:
            tips = [
                "Stay socially and physically active.",
                "Follow your doctor's advice on medications.",
                "Eat heart-healthy foods and limit salt.",
                "Monitor your blood pressure, cholesterol, and blood sugar."
            ]
    return render_template('health_tips.html', tips=tips, age=age)

@app.route('/risk_assessment', methods=['GET', 'POST'])
@login_required
@patient_required
def risk_assessment():
    print(f"Risk Assessment Route - Method: {request.method}")  # Debug log
    print(f"Request path: {request.path}")  # Debug log
    print(f"Request URL: {request.url}")  # Debug log
    
    # Always get missing_fields from upload session if present
    missing_fields = session.get('upload_missing_fields', [])
    
    if request.method == 'POST':
        print("=== RISK ASSESSMENT POST REQUEST ===")  # Debug log
        print(f"Form data keys: {list(request.form.keys())}")  # Debug log
        print(f"Form data: {dict(request.form)}")  # Debug log
        try:
            print("Processing POST request")  # Debug log
            # Get input data from form
            input_data = {}
            errors = []
            
            # Define only the features for quick assessment
            quick_assessment_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'restecg']
            
            # Define field types and validation ranges
            field_types = {
                'age': float,
                'sex': int,
                'trestbps': float,
                'chol': float,
                'thalach': float,
                'oldpeak': float,
                'restecg': int
            }
            
            validation_ranges = {
                'age': (20, 100),
                'sex': (0, 1),
                'trestbps': (90, 200),
                'chol': (100, 600),
                'thalach': (60, 200),
                'oldpeak': (0, 6.2),
                'restecg': (0, 2)
            }
            
            print(f"Form data before validation: {dict(request.form)}")  # Debug log
            
            for feature in quick_assessment_features:
                value = request.form.get(feature, '').strip()
                if not value:  # Check if field is empty
                    errors.append(f"{feature.replace('_', ' ').title()} is required")
                    continue
                try:
                    # Convert value using appropriate type
                    converter = field_types[feature]
                    converted_value = converter(float(value))  # First convert to float to handle decimal strings
                    
                    # Check if value is within valid range
                    min_val, max_val = validation_ranges[feature]
                    if not (min_val <= converted_value <= max_val):
                        errors.append(f"{feature.replace('_', ' ').title()} must be between {min_val} and {max_val}")
                        continue
                    
                    input_data[feature] = converted_value
                    print(f"Validated {feature} = {input_data[feature]} ({type(input_data[feature])})")  # Debug log
                    
                except ValueError:
                    errors.append(f"Invalid value for {feature.replace('_', ' ').title()}")
                    
            print(f"Input data after validation: {input_data}")  # Debug log
            
            # If there are any errors, return to form with error messages
            if errors:
                flash_message('Please correct the following errors:', 'error', page_specific=True)
                for error in errors:
                    flash_message(error, 'error', page_specific=True)
                return render_template('risk_assessment_form.html', 
                                     feature_descriptions=feature_descriptions,
                                     form_data=request.form,
                                     prefilled_data=session.get('prefilled_data', {}),
                                     missing_fields=missing_fields,
                                     warnings=session.get('warnings', []),
                                     form_action=url_for('risk_assessment'))  # Pass form data back to preserve valid inputs
            
            print(f"Input data: {input_data}")
            
            # Ensure all features expected by the model are present, fill with defaults if not provided by quick assessment
            # This is crucial if your model was trained on all features and expects them
            full_input_data = {
                'age': input_data.get('age'),
                'sex': input_data.get('sex'),
                'cp': float(request.form.get('cp', 0)),
                'trestbps': input_data.get('trestbps'),
                'chol': input_data.get('chol'),
                'fbs': float(request.form.get('fbs', 0)),
                'restecg': input_data.get('restecg'),  # Use the value from input_data if available
                'thalach': input_data.get('thalach'),
                'exang': float(request.form.get('exang', 0)),
                'oldpeak': input_data.get('oldpeak'),
                'slope': float(request.form.get('slope', 0)),
                'ca': float(request.form.get('ca', 0)),
                'thal': float(request.form.get('thal', 1))
            }
            
            # Only set default for restecg if it's not in input_data
            if full_input_data['restecg'] is None:
                full_input_data['restecg'] = float(request.form.get('restecg', 0))
            
            # Filter out None values and convert to dictionary required by predict_heart_disease_multi_model
            final_input_data = {k: v for k, v in full_input_data.items() if v is not None}
            
            # Make prediction using multiple models
            consensus, individual_predictions = predict_heart_disease_multi_model(final_input_data, 'models')
            
            # Validate prediction results
            if not consensus or 'probability' not in consensus:
                raise ValueError("Invalid prediction results received from models")
            
            # Ensure probability is between 0 and 1
            probability = max(0.0, min(1.0, consensus['probability']))
            
            # Determine risk level and description based on probability
            if probability >= 0.7:
                risk_level = 'High'
                risk_description = "High risk of heart disease. Immediate medical consultation is recommended."
                risk_type = "Critical"
            elif probability >= 0.4:
                risk_level = 'Medium'
                risk_description = "Medium risk of heart disease. Regular check-ups advised."
                risk_type = "Moderate"
            else:
                risk_level = 'Low'
                risk_description = "Low risk of heart disease. Maintain a healthy lifestyle."
                risk_type = "Minimal"
            
            # Update consensus with risk level, type, and description
            consensus['risk_level'] = risk_level
            consensus['risk_type'] = risk_type
            consensus['probability'] = probability
            consensus['risk_description'] = risk_description

            # Create prediction_text for the template
            prediction_text = f"Risk Level: {risk_level} ({probability:.1%})" 
            
            print(f"Prediction: {consensus}")  # Debug log
            print(f"Individual predictions: {individual_predictions}")  # Debug log
            
            # Extract individual model probabilities for the template (always calculate these)
            knn_probability = individual_predictions.get('knn', {}).get('probability', 0.0) * 100
            rf_probability = individual_predictions.get('random_forest', {}).get('probability', 0.0) * 100
            xgb_probability = individual_predictions.get('xgboost', {}).get('probability', 0.0) * 100
            
            # Generate feature importance data
            try:
                # Get feature importance from all models
                feature_importance = get_feature_importance('models')
                neighbors_data = get_neighbors_data(final_input_data)
                
                # Create visualizations directly
                print("Creating risk_factor_plot...")
                risk_factor_plot = create_feature_impact_chart(final_input_data, feature_importance)
                if risk_factor_plot is None:
                    print("ERROR: risk_factor_plot is None")
                else:
                    print("SUCCESS: risk_factor_plot created")
                risk_factor_img = convert_plot_to_base64(risk_factor_plot)
                print(f"DEBUG: risk_factor_img length: {len(risk_factor_img) if risk_factor_img else 0}")
                
                print("Creating key_risk_factors_chart...")
                key_risk_factors_chart = create_key_risk_factors_chart(final_input_data, feature_importance)
                if key_risk_factors_chart is None:
                    print("ERROR: key_risk_factors_chart is None")
                else:
                    print("SUCCESS: key_risk_factors_chart created")
                key_risk_factors_img = convert_plot_to_base64(key_risk_factors_chart)
                print(f"DEBUG: key_risk_factors_img length: {len(key_risk_factors_img) if key_risk_factors_img else 0}")
                
                # Generate comprehensive explanation with all visualizations
                explanation = generate_explanation(
                    final_input_data,
                    consensus, 
                    neighbors_data,
                    feature_importance,
                    individual_predictions
                )
                
                # Update explanation with visualizations and risk information
                explanation.update({
                    'risk_factor_img': risk_factor_img,
                    'key_risk_factors_img': key_risk_factors_img,
                    'risk_level': risk_level,
                    'risk_type': risk_type,
                    'probability': probability,
                    'risk_description': risk_description,
                    'detailed_explanation': f"Based on the analysis of your health data, you have a {risk_level.lower()} risk of heart disease with a probability of {probability:.1%}. {risk_description}"
                })
                
                # Get neighbors data for KNN explanation
                try:
                    neighbors_data = get_neighbors_data(final_input_data)
                    explanation['model_insights'] = {
                        'knn': generate_knn_explanation(final_input_data, consensus, neighbors_data, None),
                        'random_forest': generate_random_forest_explanation(final_input_data, consensus, None),
                        'xgboost': generate_xgboost_explanation(final_input_data, consensus, None)
                    }
                except Exception as e:
                    print(f"Error getting neighbors data: {str(e)}")
                    explanation['model_insights'] = {
                        'knn': generate_knn_explanation(final_input_data, consensus, None, None),
                        'random_forest': generate_random_forest_explanation(final_input_data, consensus, None),
                        'xgboost': generate_xgboost_explanation(final_input_data, consensus, None)
                    }
                
                # Store the assessment data
                try:
                    session_id = str(uuid.uuid4())
                    print(f"DEBUG: Generated session_id in risk assessment: {session_id}")  # Debug log
                    
                    if not os.path.exists('temp'):
                        os.makedirs('temp')
                    
                    assessment_data = {
                        'input_data': final_input_data,
                        'prediction': consensus,
                        'individual_predictions': individual_predictions,
                        'explanation': explanation,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save to pickle file for download functionality
                    temp_file_path = f'temp/prediction_{session_id}.pkl'
                    pd.to_pickle(assessment_data, temp_file_path)
                    print(f"DEBUG: Saved prediction data to: {temp_file_path}")  # Debug log
                    
                    # Determine the source based on session data
                    # Check if data came from PDF upload
                    if session.get('upload_extracted_fields'):
                        source = 'pdf_upload'
                        print(f"DEBUG: Setting source as 'pdf_upload' for session_id: {session_id}")
                    else:
                        source = 'manual_entry'
                        print(f"DEBUG: Setting source as 'manual_entry' for session_id: {session_id}")
                    
                    # Save to database (using UserPrediction for logged-in users)
                    save_user_prediction_to_db(current_user.id, session_id, final_input_data, consensus, individual_predictions, explanation, source)
                    
                    # Add the session_id to the explanation
                    explanation['id'] = session_id
                    
                except Exception as e:
                    print(f"Error saving assessment data: {str(e)}")
                    traceback.print_exc()
                    flash('Warning: Unable to save assessment data.', 'warning')
                    session_id = str(uuid.uuid4())  # Generate ID anyway for template
                
                # Clear session data after successful processing
                session.pop('prefilled_data', None)
                session.pop('missing_fields', None)
                session.pop('warnings', None)
                session.pop('upload_extracted_fields', None)  # Clear upload session data
                session.pop('upload_missing_fields', None)
                session.pop('upload_warnings', None)
                session.pop('upload_prefilled_data', None)
                
                # Render the results page
                return render_template('result.html',
                    explanation=explanation,
                    input_data=input_data,
                    feature_descriptions=feature_descriptions,
                    session_id=session_id,
                    prediction_text=prediction_text,
                    prediction=consensus,
                    individual_predictions=individual_predictions,
                    knn_probability=knn_probability,
                    rf_probability=rf_probability,
                    xgb_probability=xgb_probability,
                    cache_buster=datetime.now().timestamp()
                )
                
            except Exception as e:
                print(f"Error generating explanation: {str(e)}")
                traceback.print_exc()
                flash('An error occurred while generating the explanation. Basic prediction results are shown below.', 'warning')
                return render_template('result.html',
                    prediction=consensus,
                    input_data=input_data,
                    feature_descriptions=feature_descriptions,
                    prediction_text=prediction_text,
                    individual_predictions=individual_predictions,
                    knn_probability=knn_probability,
                    rf_probability=rf_probability,
                    xgb_probability=xgb_probability,
                    cache_buster=datetime.now().timestamp()
                )
            
        except Exception as e:
            print(f"Error in risk assessment: {str(e)}")  # Debug log
            flash_message(f'An error occurred: {str(e)}', 'error', page_specific=True)
            return render_template('risk_assessment_form.html', 
                                 feature_descriptions=feature_descriptions,
                                 form_data=request.form,
                                 prefilled_data=session.get('prefilled_data', {}),
                                 missing_fields=missing_fields,
                                 warnings=session.get('warnings', []),
                                 form_action=url_for('risk_assessment'))
    
    # GET request - show the assessment form
    print("Processing GET request")  # Debug log
    
    # Get prefilled data from session
    prefilled_data = session.get('prefilled_data', {})
    print(f"\nDEBUG: Raw session data: {session.get('prefilled_data')}")  # Debug log
    
    # Ensure numeric fields are properly formatted
    numeric_fields = {
        'age': float,
        'sex': int,
        'trestbps': float,
        'chol': float,
        'thalach': float,
        'oldpeak': float,
        'restecg': int,
        'ca': int  # Ensure ca is formatted as an integer string
    }
    
    print(f"DEBUG: Before formatting - prefilled_data: {prefilled_data}")  # Debug log
    for field, converter in numeric_fields.items():
        if field in prefilled_data:
            try:
                value = converter(prefilled_data[field])
                prefilled_data[field] = str(value)  # Convert to string for template
                print(f"DEBUG: Formatted {field} = {prefilled_data[field]} (type: {type(prefilled_data[field])})")  # Debug log
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Error formatting {field}: {str(e)}")  # Debug log
                prefilled_data.pop(field, None)  # Remove invalid value
    
    print(f"DEBUG: After formatting - prefilled_data: {prefilled_data}")  # Debug log
    
    # After formatting prefilled_data and before rendering the template in risk_assessment GET
    if 'ca' in prefilled_data and prefilled_data['ca'] is not None:
        try:
            prefilled_data['ca'] = str(int(float(prefilled_data['ca'])))
        except Exception:
            prefilled_data['ca'] = None
    
    # Render the form with prefilled data
    return render_template('risk_assessment_form.html',
                         feature_descriptions=feature_descriptions,
                         prefilled_data=prefilled_data,
                         missing_fields=missing_fields,
                         warnings=session.get('warnings', []),
                         form_action=url_for('risk_assessment'))

@app.route('/assessment')
@login_required
@patient_required
def assessment():
    # Clear any existing session data to ensure fresh start
    session.pop('prefilled_data', None)
    session.pop('upload_missing_fields', None)
    session.pop('warnings', None)
    session.pop('chat_session_id', None)
    session.pop('chat_state', None)
    session.pop('chat_data', None)
    
    # Get prefilled data from query parameters only (not from session)
    prefilled_data = {
        'chol': request.args.get('chol', type=float),
        'trestbps': request.args.get('trestbps', type=float),
        'thalach': request.args.get('thalach', type=float),
        'oldpeak': request.args.get('oldpeak', type=float)
    }
    
    # Remove None values
    prefilled_data = {k: v for k, v in prefilled_data.items() if v is not None}
    
    # Get list of missing fields
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    missing_fields = [field for field in required_fields if field not in prefilled_data]
    
    return render_template('assessment.html', 
                         feature_descriptions=feature_descriptions,
                         prefilled_data=prefilled_data,
                         missing_fields=missing_fields)

@app.route('/clear_assessment', methods=['POST'])
@login_required
@patient_required
def clear_assessment():
    """Clear all assessment-related session data"""
    try:
        # Clear all assessment-related session data
        session.pop('prefilled_data', None)
        session.pop('upload_missing_fields', None)
        session.pop('warnings', None)
        session.pop('chat_session_id', None)
        session.pop('chat_state', None)
        session.pop('chat_data', None)
        session.pop('chatbot_stage', None)
        session.pop('chatbot_data', None)
        session.pop('chatbot_symptom_details', None)
        session.pop('chatbot_exchange_count', None)
        session.pop('chatbot_conversation_context', None)
        session.pop('chatbot_consultation_complete', None)
        
        return jsonify({'success': True, 'message': 'Assessment data cleared'})
    except Exception as e:
        print(f"Error clearing assessment data: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to clear assessment data'}), 500

@app.route('/chat_assessment', methods=['GET', 'POST'])
@login_required
@patient_required
def chat_assessment():
    """Professional chat-based cardiovascular assessment using enhanced AI doctor."""
    try:
        if request.method == 'POST':
            data = request.get_json()
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return jsonify({
                    'error': 'Please provide a message.'
                }), 400
            
            # Get or create chatbot instance in session
            session_id = session.get('chat_session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                session['chat_session_id'] = session_id
                # Initialize the AI-enhanced professional chatbot
                from src.gemini_chatbot import GeminiHeartDoctorChatbot
                chatbot = GeminiHeartDoctorChatbot()
                # Configure API key if available
                api_key = os.environ.get('GEMINI_API_KEY')
                if api_key:
                    chatbot.configure_api(api_key)
                session['chatbot_stage'] = chatbot.current_stage
                session['chatbot_data'] = chatbot.consultation_data
                session['chatbot_symptom_details'] = chatbot.symptom_details
                session['chatbot_exchange_count'] = chatbot.exchange_count
                session['chatbot_conversation_context'] = chatbot.conversation_context
                session['chatbot_consultation_complete'] = chatbot.consultation_complete
            else:
                # Restore chatbot state from session
                from src.gemini_chatbot import GeminiHeartDoctorChatbot
                chatbot = GeminiHeartDoctorChatbot()
                # Configure API key if available
                api_key = os.environ.get('GEMINI_API_KEY')
                if api_key:
                    chatbot.configure_api(api_key)
                chatbot.current_stage = session.get('chatbot_stage', 'initial_greeting')
                chatbot.consultation_data = session.get('chatbot_data', {})
                chatbot.symptom_details = session.get('chatbot_symptom_details', {})
                chatbot.exchange_count = session.get('chatbot_exchange_count', 0)
                chatbot.conversation_context = session.get('chatbot_conversation_context', [])
                chatbot.consultation_complete = session.get('chatbot_consultation_complete', False)
                # Update conversation stage based on restored exchange count
                chatbot.update_conversation_stage()
            
            # Process the user message using the professional chatbot
            response, completed_data = chatbot.process_response(user_message)
            
            # Debug logging
            print(f"DEBUG: Exchange count: {chatbot.exchange_count}, Stage: {chatbot.current_stage}, Completed: {chatbot.consultation_complete}")
            
            # Analyze symptoms for debugging
            from src.gemini_chatbot import GeminiHeartDoctorChatbot
            temp_chatbot = GeminiHeartDoctorChatbot()
            symptoms = temp_chatbot.analyze_symptoms_flexible(user_message)
            if symptoms:
                print(f"DEBUG: Detected symptoms: {symptoms}")
                for symptom, details in symptoms.items():
                    print(f"DEBUG: {symptom} - Severity: {details['severity']}")
            
            # Check emergency and urgent detection
            is_emergency = temp_chatbot.detect_emergency(user_message)
            is_urgent = temp_chatbot.detect_urgent(user_message)
            print(f"DEBUG: Emergency detected: {is_emergency}, Urgent detected: {is_urgent}")
            
            # Update session with chatbot state
            session['chatbot_stage'] = chatbot.current_stage
            session['chatbot_data'] = chatbot.consultation_data
            session['chatbot_symptom_details'] = chatbot.symptom_details
            session['chatbot_exchange_count'] = chatbot.exchange_count
            session['chatbot_conversation_context'] = chatbot.conversation_context
            session['chatbot_consultation_complete'] = chatbot.consultation_complete
            
            # Determine input help based on current stage
            stage_input_help = {
                'initial_greeting': 'Please take your time and describe what brings you here today',
                'chief_complaint': 'Describe your main symptom or concern in detail',
                'symptom_analysis': 'Please describe any symptoms you are experiencing',
                'symptom_details': 'Provide specific details about your symptoms',
                'medical_history': 'Share your medical history and current medications',
                'risk_factors': 'Tell me about your lifestyle and family history',
                'assessment_summary': 'Thank you for the comprehensive information',
                'recommendations': 'Feel free to ask any questions about the assessment',
                'completed': 'Consultation completed. Click "New Assessment" to start fresh.'
            }
            
            input_placeholder = 'Type your response here...'
            input_help = stage_input_help.get(chatbot.current_stage, 'Please provide your response')
            
            # Save completed consultation data if assessment is finished
            if completed_data and chatbot.current_stage == 'completed':
                try:
                    # Save the comprehensive consultation to database or temp storage
                    consultation_id = str(uuid.uuid4())
                    consultation_data = {
                        'user_id': current_user.id,
                        'consultation_id': consultation_id,
                        'consultation_data': completed_data,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'professional_chat_assessment'
                    }
                    
                    # Save to temp directory for potential report generation
                    if not os.path.exists('temp'):
                        os.makedirs('temp')
                    
                    temp_file_path = f'temp/consultation_{consultation_id}.pkl'
                    pd.to_pickle(consultation_data, temp_file_path)
                    
                except Exception as save_error:
                    print(f"Error saving consultation data: {str(save_error)}")
                    # Continue even if saving fails
            
            return jsonify({
                'bot_message': response,
                'success': True,
                'current_stage': chatbot.current_stage,
                'input_placeholder': input_placeholder,
                'input_help': input_help,
                'completed': chatbot.current_stage == 'completed',
                'exchange_count': chatbot.exchange_count,
                'debug_info': f"Stage: {chatbot.current_stage}, Exchange: {chatbot.exchange_count}"
            })
        
        # GET request - Initialize new chat session
        session.pop('chat_session_id', None)
        session.pop('chatbot_stage', None)
        session.pop('chatbot_data', None)
        session.pop('chatbot_symptom_details', None)
        session.pop('chatbot_exchange_count', None)
        session.pop('chatbot_conversation_context', None)
        session.pop('chatbot_consultation_complete', None)
        
        return render_template('chat_assessment.html')
    
    except Exception as e:
        print(f"Error in chat_assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        # Reset session data on error
        session.pop('chat_session_id', None)
        session.pop('chatbot_stage', None)
        session.pop('chatbot_data', None)
        session.pop('chatbot_symptom_details', None)
        session.pop('chatbot_exchange_count', None)
        session.pop('chatbot_conversation_context', None)
        session.pop('chatbot_consultation_complete', None)
        return jsonify({
            'error': 'I apologize for the technical difficulty. Let me restart our consultation to ensure I can help you properly.',
            'reset': True
        }), 500

# Old chat functions removed - now using ProfessionalHeartDoctorChatbot

@app.route('/chat_assessment_greeting', methods=['GET'])
@login_required
@patient_required
def chat_assessment_greeting():
    """Get the initial greeting for the professional chat assessment."""
    try:
        from src.gemini_chatbot import GeminiHeartDoctorChatbot
        chatbot = GeminiHeartDoctorChatbot()
        # Configure API key if available
        api_key = os.environ.get('GEMINI_API_KEY')
        ai_configured = False
        if api_key:
            ai_configured = chatbot.configure_api(api_key)
        
        greeting = chatbot.get_initial_greeting()
        
        return jsonify({
            'success': True,
            'greeting': greeting,
            'ai_configured': ai_configured
        })
        
    except Exception as e:
        print(f"Error getting chat greeting: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Unable to load greeting'
        }), 500



@app.route('/clear_extraction', methods=['POST'])
def clear_extraction():
    # Clear session variables
    session.pop('extracted_data', None)
    session.pop('missing_fields', None)
    return jsonify(success=True)
@app.route('/explanation')
def explanation():
    # Find the most recent prediction file
    recent_session_id = None
    temp_dir = 'temp'
    if os.path.exists(temp_dir):
        prediction_files = [f for f in os.listdir(temp_dir) if f.startswith('prediction_')]
        if prediction_files:
            # Get the most recent prediction file
            recent_file = max(prediction_files, key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)))
            recent_session_id = recent_file.replace('prediction_', '').replace('.pkl', '')
    if recent_session_id:
        return redirect(url_for('explain_prediction', session_id=recent_session_id))
    else:
        flash('Please make a prediction first to view the explanation.', 'info')
        return redirect(url_for('index'))

@app.route('/quick_assessment', methods=['GET', 'POST'])
@login_required
@patient_required
def quick_assessment():
    """Enhanced Quick Assessment with multiple pathways"""
    if request.method == 'GET':
        # Return the quick assessment interface
        return render_template('quick_assessment.html', 
                             feature_descriptions=feature_descriptions)
    
    try:
        # Handle different assessment types
        assessment_type = request.form.get('assessment_type', 'form')
        
        if assessment_type == 'symptom':
            return handle_symptom_assessment()
        elif assessment_type == 'voice':
            return handle_voice_assessment()
        elif assessment_type == 'emergency':
            return handle_emergency_assessment()
        else:
            return handle_form_assessment()
            
    except Exception as e:
        print(f"Error in quick_assessment: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your assessment.',
            'details': str(e)
        }), 500

def handle_symptom_assessment():
    """Handle symptom-based quick assessment"""
    try:
        symptoms = request.form.get('symptoms', '').strip().lower()
        age = request.form.get('age', type=int)
        sex = request.form.get('sex', type=int)
        
        if not symptoms or not age or sex is None:
            return jsonify({
                'error': 'Please provide symptoms, age, and sex information.'
            }), 400
        
        # Analyze symptoms for emergency indicators
        emergency_keywords = [
            'severe chest pain', 'crushing pain', 'radiating pain',
            'difficulty breathing', 'shortness of breath', 'can\'t breathe',
            'sweating profusely', 'nausea', 'dizziness', 'fainting',
            'heart racing', 'palpitations', 'irregular heartbeat'
        ]
        
        high_risk_keywords = [
            'chest discomfort', 'chest tightness', 'pressure in chest',
            'arm pain', 'jaw pain', 'back pain', 'fatigue', 'weakness'
        ]
        
        # Assess symptom severity
        is_emergency = any(keyword in symptoms for keyword in emergency_keywords)
        is_high_risk = any(keyword in symptoms for keyword in high_risk_keywords)
        
        if is_emergency:
            return jsonify({
                'assessment_type': 'emergency',
                'risk_level': 'EMERGENCY',
                'message': '⚠️ EMERGENCY: Your symptoms may indicate a heart attack. Call 911 immediately!',
                'recommendations': [
                    'Call emergency services (911) NOW',
                    'Chew aspirin if not allergic (unless told otherwise)',
                    'Sit or lie down with head elevated',
                    'Stay calm and wait for help',
                    'Have someone stay with you'
                ],
                'emergency_numbers': ['911', '112'],
                'completed': True
            })
        
        # Generate estimated risk based on symptoms and basic demographics
        base_risk = 0.2  # Base risk
        
        # Age factor
        if age > 65:
            base_risk += 0.2
        elif age > 45:
            base_risk += 0.1
        
        # Sex factor (males higher risk)
        if sex == 1:
            base_risk += 0.1
        
        # Symptom factors
        if is_high_risk:
            base_risk += 0.3
        
        # Additional symptom analysis
        if 'chest' in symptoms:
            base_risk += 0.2
        if 'fatigue' in symptoms or 'tired' in symptoms:
            base_risk += 0.1
        if 'smoking' in symptoms or 'smoke' in symptoms:
            base_risk += 0.2
        
        risk_probability = min(base_risk, 0.95)  # Cap at 95%
        
        # Determine risk level
        if risk_probability >= 0.7:
            risk_level = 'High'
            risk_description = 'High risk indicators detected. Medical consultation recommended within 24 hours.'
        elif risk_probability >= 0.4:
            risk_level = 'Medium'
            risk_description = 'Moderate risk detected. Schedule appointment with healthcare provider.'
        else:
            risk_level = 'Low'
            risk_description = 'Low risk based on symptoms. Monitor symptoms and maintain healthy lifestyle.'
        
        # Generate recommendations
        recommendations = generate_symptom_recommendations(symptoms, risk_level, age, sex)
        
        return jsonify({
            'assessment_type': 'symptom',
            'risk_level': risk_level,
            'probability': risk_probability,
            'message': f'Risk Assessment: {risk_level} ({risk_probability:.1%})',
            'description': risk_description,
            'recommendations': recommendations,
            'next_steps': [
                'Consider a comprehensive assessment for more accurate results',
                'Monitor symptoms and seek medical attention if they worsen',
                'Maintain a symptom diary for healthcare providers'
            ],
            'completed': True
        })
        
    except Exception as e:
        print(f"Error in symptom assessment: {str(e)}")
        return jsonify({'error': 'Error processing symptom assessment'}), 500

def handle_form_assessment():
    """Handle quick form-based assessment"""
    try:
        # Get essential fields for quick assessment
        essential_fields = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']
        input_data = {}
        missing_fields = []
        
        for field in essential_fields:
            value = request.form.get(field)
            if value and value.strip():
                try:
                    input_data[field] = float(value)
                except ValueError:
                    missing_fields.append(field)
            else:
                missing_fields.append(field)
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'missing_fields': missing_fields
            }), 400
        
        # Use only the quick assessment model and features
        consensus = predict_quick_assessment(input_data, 'models')
        individual_predictions = {'quick_assessment': consensus}
        
        # Generate session ID for detailed results
        session_id = str(uuid.uuid4())
        print(f"DEBUG: Generated session_id: {session_id}")  # Debug log
        
        # Generate feature importance data and visualizations for quick assessment
        try:
            # Get feature importance from the quick assessment model
            feature_importance = get_feature_importance('models')
            neighbors_data = get_neighbors_data(input_data)
            
            # Create visualizations for quick assessment
            print("Creating risk_factor_plot for quick assessment...")
            risk_factor_plot = create_feature_impact_chart(input_data, feature_importance)
            risk_factor_img = convert_plot_to_base64(risk_factor_plot) if risk_factor_plot else None
            
            print("Creating key_risk_factors_chart for quick assessment...")
            key_risk_factors_chart = create_key_risk_factors_chart(input_data, feature_importance)
            key_risk_factors_img = convert_plot_to_base64(key_risk_factors_chart) if key_risk_factors_chart else None
            
            # Generate comprehensive explanation with visualizations
            explanation = generate_explanation(
                input_data,
                consensus, 
                neighbors_data,
                feature_importance,
                individual_predictions
            )
            
            # Update explanation with visualizations and assessment type
            explanation.update({
                'risk_factor_img': risk_factor_img,
                'key_risk_factors_img': key_risk_factors_img,
                'assessment_type': 'quick_form',
                'id': session_id
            })
            
        except Exception as e:
            print(f"Error generating visualizations for quick assessment: {str(e)}")
            # Fallback to basic explanation if visualization fails
            explanation = {
                'risk_level': consensus.get('risk_level', 'Unknown'),
                'probability': consensus.get('probability', 0.0),
                'risk_description': consensus.get('risk_description', ''),
                'assessment_type': 'quick_form',
                'id': session_id
            }
        
        # Save assessment data
        if not os.path.exists('temp'):
            os.makedirs('temp')
            
        temp_data = {
            'input_data': input_data,
            'prediction': consensus,
            'individual_predictions': individual_predictions,
            'explanation': explanation,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'quick_assessment'
        }
        
        temp_file_path = f'temp/prediction_{session_id}.pkl'
        pd.to_pickle(temp_data, temp_file_path)
        print(f"DEBUG: Saved prediction data to: {temp_file_path}")  # Debug log
        print(f"DEBUG: Session ID: {session_id}")  # Debug log
        
        # Save to database (using UserPrediction for logged-in users)
        save_user_prediction_to_db(current_user.id, session_id, input_data, consensus, individual_predictions, explanation, 'quick_assessment')
        
        return jsonify({
            'assessment_type': 'form',
            'risk_level': consensus.get('risk_level', 'Unknown'),
            'probability': consensus.get('probability', 0.0),
            'message': f"Risk Level: {consensus.get('risk_level', 'Unknown')} ({consensus.get('probability', 0.0):.1%})",
            'description': consensus.get('risk_description', ''),
            'session_id': session_id,
            'detailed_url': url_for('explain_prediction', session_id=session_id),
            'individual_predictions': {
                'quick_assessment': consensus.get('probability', 0.0)
            },
            'recommendations': (
                [
                    'Schedule an immediate appointment with your healthcare provider.',
                    'Monitor your blood pressure and heart rate daily.',
                    'Keep a symptom diary to track any chest pain or discomfort.'
                ] if consensus.get('risk_level') == 'High' else
                [
                    'Continue regular physical activity.',
                    'Eat a balanced, heart-healthy diet.',
                    'Avoid smoking and limit alcohol consumption.',
                    'Manage stress and get regular sleep.'
                ] if consensus.get('risk_level') == 'Low' else
                [
                    'Schedule a follow-up with your healthcare provider within a week.',
                    'Start monitoring your vital signs regularly.',
                    'Review your current medications with your doctor.'
                ]
            ),
            'completed': True
        })
        
    except Exception as e:
        print(f"Error in form assessment: {str(e)}")
        return jsonify({'error': 'Error processing form assessment'}), 500

def handle_emergency_assessment():
    """Handle emergency assessment pathway"""
    try:
        symptoms = request.form.get('emergency_symptoms', '').strip().lower()
        
        emergency_response = {
            'assessment_type': 'emergency',
            'risk_level': 'EMERGENCY',
            'message': '🚨 EMERGENCY PROTOCOL ACTIVATED',
            'immediate_actions': [
                '1. Call 911 immediately',
                '2. Chew 1 aspirin (unless allergic)',
                '3. Sit down with back support',
                '4. Loosen tight clothing',
                '5. Stay calm and breathe slowly'
            ],
            'while_waiting': [
                'Have someone stay with you',
                'Prepare list of medications',
                'Note time symptoms started',
                'Keep doors unlocked for paramedics',
                'Gather insurance/ID cards'
            ],
            'warning_signs': [
                'Severe chest pain or pressure',
                'Pain radiating to arm, neck, jaw',
                'Difficulty breathing',
                'Cold sweats, nausea',
                'Irregular heartbeat'
            ],
            'emergency_numbers': ['911', '112'],
            'completed': True
        }
        
        return jsonify(emergency_response)
        
    except Exception as e:
        print(f"Error in emergency assessment: {str(e)}")
        return jsonify({'error': 'Error processing emergency assessment'}), 500

def handle_voice_assessment():
    """Handle voice-based assessment (placeholder for future implementation)"""
    try:
        # This would integrate with speech recognition APIs
        # For now, return a placeholder response
        return jsonify({
            'assessment_type': 'voice',
            'message': 'Voice assessment feature coming soon. Please use the form or symptom assessment.',
            'error': 'Voice assessment not yet implemented'
        }), 501
        
    except Exception as e:
        print(f"Error in voice assessment: {str(e)}")
        return jsonify({'error': 'Error processing voice assessment'}), 500

def generate_symptom_recommendations(symptoms, risk_level, age, sex):
    """Generate personalized recommendations based on symptoms"""
    recommendations = []
    
    # General recommendations based on risk level
    if risk_level == 'High':
        recommendations.extend([
            'Schedule immediate medical consultation (within 24 hours)',
            'Monitor symptoms closely and seek emergency care if they worsen',
            'Avoid strenuous activities until cleared by a doctor',
            'Keep emergency medications readily available'
        ])
    elif risk_level == 'Medium':
        recommendations.extend([
            'Schedule appointment with healthcare provider within 1-2 weeks',
            'Monitor blood pressure and heart rate if possible',
            'Maintain a symptom diary',
            'Make lifestyle modifications (diet, exercise, stress management)'
        ])
    else:
        recommendations.extend([
            'Continue regular health check-ups',
            'Maintain heart-healthy lifestyle',
            'Monitor symptoms and track any changes',
            'Stay active with appropriate exercise'
        ])
    
    # Symptom-specific recommendations
    if 'chest' in symptoms:
        recommendations.append('Note characteristics of chest pain (location, duration, triggers)')
    
    if 'shortness' in symptoms or 'breathing' in symptoms:
        recommendations.append('Avoid triggers and consider pulmonary function evaluation')
    
    if 'fatigue' in symptoms or 'tired' in symptoms:
        recommendations.append('Ensure adequate sleep and consider blood work evaluation')
    
    # Age and sex specific recommendations
    if age > 65:
        recommendations.append('Consider comprehensive geriatric cardiovascular assessment')
    
    if sex == 1 and age > 45:  # Males over 45
        recommendations.append('Regular cholesterol and blood pressure monitoring essential')
    elif sex == 0 and age > 55:  # Females over 55
        recommendations.append('Post-menopausal heart health monitoring recommended')
    
    return recommendations

@app.route('/download_report/<session_id>', methods=['GET'])
@login_required
@patient_required
def download_report(session_id):
    try:
        print(f"DEBUG: Attempting to download report for session_id: {session_id}")  # Debug log
        temp_file = f'temp/prediction_{session_id}.pkl'
        print(f"DEBUG: Looking for prediction file: {temp_file}")  # Debug log
        
        if not os.path.exists(temp_file):
            print(f"ERROR: Prediction data not found for session_id: {session_id} at {temp_file}")  # Debug log
            flash('Report not found. Please make a new assessment.', 'danger')
            return redirect(url_for('index'))
        
        # Load data using pandas (since it was saved with pd.to_pickle)
        data = pd.read_pickle(temp_file)
        print(f"DEBUG: Loaded data keys: {list(data.keys())}")  # Debug log
        
        # Extract the required data
        input_data = data.get('input_data', {})
        prediction = data.get('prediction', {})
        individual_predictions = data.get('individual_predictions', {})
        explanation = data.get('explanation', {})
        
        print(f"DEBUG: Input data keys: {list(input_data.keys()) if input_data else 'None'}")
        print(f"DEBUG: Prediction keys: {list(prediction.keys()) if prediction else 'None'}")
        print(f"DEBUG: Explanation keys: {list(explanation.keys()) if explanation else 'None'}")
        
        # Generate visualizations and explanation for the PDF
        try:
            feature_importance = get_feature_importance()
            neighbors_data = get_neighbors_data(input_data)
            
            # Create plots using the same functions as the main application
            print("Creating charts for PDF download...")
            
            # Create the same charts as the main application
            risk_factor_plot = create_feature_impact_chart(input_data, feature_importance)
            if risk_factor_plot is None:
                print("ERROR: risk_factor_plot is None in download_report")
            else:
                print("SUCCESS: risk_factor_plot created for PDF")
            risk_factor_img = convert_plot_to_base64(risk_factor_plot)
            
            key_risk_factors_chart = create_key_risk_factors_chart(input_data, feature_importance)
            if key_risk_factors_chart is None:
                print("ERROR: key_risk_factors_chart is None in download_report")
            else:
                print("SUCCESS: key_risk_factors_chart created for PDF")
            key_risk_factors_img = convert_plot_to_base64(key_risk_factors_chart)
            
            # Generate explanation text and data with model-specific explanations
            explanation = generate_explanation(
                input_data, 
                prediction, 
                neighbors_data, 
                feature_importance,
                individual_predictions
            )
            
            # Update explanation with the correct chart images
            explanation.update({
                'risk_factor_img': risk_factor_img,
                'key_risk_factors_img': key_risk_factors_img
            })
            
            print(f"DEBUG: PDF explanation keys: {list(explanation.keys())}")
            print(f"DEBUG: risk_factor_img length: {len(explanation.get('risk_factor_img', '')) if explanation.get('risk_factor_img') else 0}")
            print(f"DEBUG: key_risk_factors_img length: {len(explanation.get('key_risk_factors_img', '')) if explanation.get('key_risk_factors_img') else 0}")
            
        except Exception as viz_error:
            print(f"WARNING: Error generating visualizations: {viz_error}")
            import traceback
            traceback.print_exc()
            # Continue without visualizations
            explanation = explanation or {}
        
        # Generate PDF report
        pdf_content = generate_pdf_report(explanation, input_data, feature_descriptions, session_id)
        
        # Create response with PDF
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=heart_disease_report_{session_id[:8]}.pdf'
        
        print(f"DEBUG: Successfully generated and returning PDF for session_id: {session_id}")  # Debug log
        return response
        
    except Exception as e:
        print(f"ERROR generating PDF for session_id {session_id}: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Print full error traceback
        flash(f'An error occurred while generating the report: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/prefilled_form')
@login_required
@patient_required
def prefilled_form():
    """Handle the prefilled form with extracted data"""
    try:
        # Get all the data passed in the URL
        extracted_data = request.args.to_dict()
        
        # Convert numeric values from strings
        for key, value in extracted_data.items():
            try:
                extracted_data[key] = float(value)
            except (ValueError, TypeError):
                # Keep as string if not a number
                pass
        
        return render_template('prefilled_form.html', 
                             extracted_data=extracted_data,
                             feature_descriptions=feature_descriptions)
    except Exception as e:
        print(f"Error in prefilled_form route: {str(e)}")
        flash(f"Error displaying form: {str(e)}", 'error')
        return redirect(url_for('upload_report'))

@app.route('/api/my_predictions', methods=['GET'])
@login_required
@patient_required
def api_my_predictions():
    limit = int(request.args.get('limit', 50))
    predictions = get_user_predictions(current_user.id, limit=limit)
    return jsonify({
        'success': True,
        'predictions': predictions,
        'count': len(predictions)
    })

@app.route('/api/my_predictions/<int:prediction_id>', methods=['DELETE'])
@login_required
@patient_required
def delete_my_prediction(prediction_id):
    try:
        prediction = UserPrediction.query.filter_by(id=prediction_id, user_id=current_user.id).first()
        if not prediction:
            return jsonify({'success': False, 'message': 'Prediction not found or not authorized'}), 404
        db.session.delete(prediction)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Prediction deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/recent_predictions', methods=['GET'])
def api_recent_predictions():
    try:
        limit = int(request.args.get('limit', 5))
        # Get from both Prediction and UserPrediction tables
        predictions = []
        # From Prediction (legacy, anonymous)
        predictions += get_all_predictions_from_db(limit=limit)
        # From UserPrediction (authenticated users)
        user_preds = UserPrediction.query.order_by(UserPrediction.timestamp.desc()).limit(limit).all()
        predictions += [pred.to_dict() for pred in user_preds]
        # Sort all by timestamp descending and take top N
        predictions = sorted(predictions, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
        return jsonify({'success': True, 'predictions': predictions, 'count': len(predictions)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Admin Routes



@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin users management page"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 20
        
        # Get search parameters
        search = request.args.get('search', '')
        role_filter = request.args.get('role', '')
        
        # Build query
        query = User.query
        
        if search:
            query = query.filter(
                db.or_(
                    User.username.contains(search),
                    User.email.contains(search),
                    User.first_name.contains(search),
                    User.last_name.contains(search)
                )
            )
        
        if role_filter:
            query = query.filter(User.role == role_filter)
        
        # Paginate results
        users = query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return render_template('admin/users.html', users=users, search=search, role_filter=role_filter)
    except Exception as e:
        print(f"Admin users error: {str(e)}")
        flash(f'Error loading users: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/admin/user/<int:user_id>')
@login_required
@admin_required
def admin_user_detail(user_id):
    """Admin user detail page"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Get user's predictions
        predictions = UserPrediction.query.filter_by(user_id=user_id).order_by(UserPrediction.timestamp.desc()).limit(10).all()
        
        # Get user's medical reports
        reports = UserMedicalReport.query.filter_by(user_id=user_id).order_by(UserMedicalReport.upload_date.desc()).limit(10).all()
        
        return render_template('admin/user_detail.html', user=user, predictions=predictions, reports=reports)
    except Exception as e:
        print(f"Admin user detail error: {str(e)}")
        flash(f'Error loading user details: {str(e)}', 'error')
        return redirect(url_for('admin_users'))

@app.route('/admin/user/<int:user_id>/toggle_status', methods=['POST'])
@login_required
@admin_required
def admin_toggle_user_status(user_id):
    """Toggle user active status"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent admin from deactivating themselves
        if user.id == current_user.id:
            flash('You cannot deactivate your own account.', 'error')
            return redirect(url_for('admin_user_detail', user_id=user_id))
        
        user.is_active = not user.is_active
        db.session.commit()
        
        status = 'activated' if user.is_active else 'deactivated'
        flash(f'User {user.username} has been {status}.', 'success')
        
        return redirect(url_for('admin_user_detail', user_id=user_id))
    except Exception as e:
        db.session.rollback()
        print(f"Toggle user status error: {str(e)}")
        flash(f'Error updating user status: {str(e)}', 'error')
        return redirect(url_for('admin_user_detail', user_id=user_id))

@app.route('/admin/user/<int:user_id>/change_role', methods=['POST'])
@login_required
@admin_required
def admin_change_user_role(user_id):
    """Change user role"""
    try:
        user = User.query.get_or_404(user_id)
        new_role = request.form.get('role')
        
        if new_role not in ['admin', 'patient']:
            flash('Invalid role specified.', 'error')
            return redirect(url_for('admin_user_detail', user_id=user_id))
        
        # Prevent admin from changing their own role
        if user.id == current_user.id:
            flash('You cannot change your own role.', 'error')
            return redirect(url_for('admin_user_detail', user_id=user_id))
        
        old_role = user.role
        user.role = new_role
        db.session.commit()
        
        flash(f'User {user.username} role changed from {old_role} to {new_role}.', 'success')
        
        return redirect(url_for('admin_user_detail', user_id=user_id))
    except Exception as e:
        db.session.rollback()
        print(f"Change user role error: {str(e)}")
        flash(f'Error changing user role: {str(e)}', 'error')
        return redirect(url_for('admin_user_detail', user_id=user_id))

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    """Delete user"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent admin from deleting themselves
        if user.id == current_user.id:
            flash('You cannot delete your own account.', 'error')
            return redirect(url_for('admin_user_detail', user_id=user_id))
        
        username = user.username
        db.session.delete(user)
        db.session.commit()
        
        flash(f'User {username} has been deleted.', 'success')
        return redirect(url_for('admin_users'))
    except Exception as e:
        db.session.rollback()
        print(f"Delete user error: {str(e)}")
        flash(f'Error deleting user: {str(e)}', 'error')
        return redirect(url_for('admin_user_detail', user_id=user_id))



@app.route('/admin/all_reports')
@login_required
@admin_required
def admin_all_reports():
    """Admin view of all reports from all users"""
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        # Get search parameters
        search = request.args.get('search', '')
        user_filter = request.args.get('user', '')
        risk_filter = request.args.get('risk_level', '')
        
        # Build query for all predictions with user info
        query = db.session.query(UserPrediction, User).join(User, UserPrediction.user_id == User.id)
        
        if search:
            query = query.filter(
                db.or_(
                    User.username.contains(search),
                    User.email.contains(search),
                    User.first_name.contains(search),
                    User.last_name.contains(search)
                )
            )
        
        if user_filter:
            query = query.filter(User.id == user_filter)
        
        if risk_filter:
            query = query.filter(UserPrediction.risk_level == risk_filter)
        
        # Get total count for pagination
        total_count = query.count()
        
        # Paginate results
        results = query.order_by(UserPrediction.timestamp.desc()).offset((page - 1) * per_page).limit(per_page).all()
        
        # Format results
        formatted_predictions = []
        for prediction, user in results:
            try:
                formatted_prediction = {
                    'id': prediction.session_id,
                    'user_id': user.id,
                    'username': user.username,
                    'user_email': user.email,
                    'user_name': user.get_full_name(),
                    'date': prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S') if prediction.timestamp else 'Unknown',
                    'risk_level': prediction.risk_level or 'Unknown',
                    'probability': int((prediction.probability or 0) * 100),
                    'source': prediction.source,
                    'metrics': json.loads(prediction.input_data) if prediction.input_data else {}
                }
                formatted_predictions.append(formatted_prediction)
            except Exception as e:
                print(f"Error formatting prediction: {str(e)}")
                continue
        
        # Get all users for filter dropdown
        all_users = User.query.order_by(User.username).all()
        
        return render_template('admin/all_reports.html', 
                             predictions=formatted_predictions,
                             page=page,
                             per_page=per_page,
                             total_predictions=total_count,
                             search=search,
                             user_filter=user_filter,
                             risk_filter=risk_filter,
                             all_users=all_users)
    
    except Exception as e:
        print(f"Error in admin all reports route: {str(e)}")
        flash(f'Error loading all reports: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/admin/view_report/<session_id>')
@login_required
@admin_required
def admin_view_report(session_id):
    """Admin view of a specific report"""
    try:
        # Get the prediction
        prediction = UserPrediction.query.filter_by(session_id=session_id).first()
        if not prediction:
            flash('Report not found.', 'error')
            return redirect(url_for('admin_all_reports'))
        
        # Get user info
        user = User.query.get(prediction.user_id)
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('admin_all_reports'))
        
        # Parse the data
        input_data = json.loads(prediction.input_data) if prediction.input_data else {}
        prediction_data = json.loads(prediction.prediction) if prediction.prediction else {}
        explanation = json.loads(prediction.explanation) if prediction.explanation else {}
        
        return render_template('explanation.html',
                             explanation=explanation,
                             input_data=input_data,
                             feature_descriptions=feature_descriptions,
                             session_id=session_id,
                             prediction=prediction_data,
                             user=user,
                             is_admin_view=True)
    
    except Exception as e:
        print(f"Error in admin view report: {str(e)}")
        flash(f'Error viewing report: {str(e)}', 'error')
        return redirect(url_for('admin_all_reports'))

@app.route('/admin/download_report/<session_id>')
@login_required
@admin_required
def admin_download_report(session_id):
    """Admin download of a specific report"""
    try:
        print(f"DEBUG: Admin attempting to download report for session_id: {session_id}")
        
        # Get the prediction from database
        prediction = UserPrediction.query.filter_by(session_id=session_id).first()
        if not prediction:
            flash('Report not found.', 'error')
            return redirect(url_for('admin_all_reports'))
        
        # Get user info
        user = User.query.get(prediction.user_id)
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('admin_all_reports'))
        
        # Parse the data from database
        input_data = json.loads(prediction.input_data) if prediction.input_data else {}
        prediction_data = json.loads(prediction.prediction) if prediction.prediction else {}
        individual_predictions = json.loads(prediction.individual_predictions) if prediction.individual_predictions else {}
        explanation = json.loads(prediction.explanation) if prediction.explanation else {}
        
        print(f"DEBUG: Admin - Input data keys: {list(input_data.keys()) if input_data else 'None'}")
        print(f"DEBUG: Admin - Prediction keys: {list(prediction_data.keys()) if prediction_data else 'None'}")
        print(f"DEBUG: Admin - Explanation keys: {list(explanation.keys()) if explanation else 'None'}")
        
        # Use the existing explanation from database, or create a simple one
        if not explanation:
            explanation = {
                'risk_level': prediction.risk_level or 'Unknown',
                'probability': prediction.probability or 0.0,
                'risk_description': f'Risk level: {prediction.risk_level or "Unknown"}',
                'explanation_text': 'Report generated from stored data.'
            }
        
        # Generate PDF report
        pdf_content = generate_pdf_report(explanation, input_data, feature_descriptions, session_id)
        
        # Create response with PDF
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Heart_Disease_Report_{user.username}_{session_id}.pdf'
        
        print(f"DEBUG: Successfully generated and returning admin PDF for session_id: {session_id}")
        return response
    
    except Exception as e:
        print(f"ERROR generating admin PDF for session_id {session_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error downloading report: {str(e)}', 'error')
        return redirect(url_for('admin_all_reports'))

@app.route('/admin/delete_report/<session_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_report(session_id):
    """Admin delete a specific report"""
    try:
        # Get the prediction
        prediction = UserPrediction.query.filter_by(session_id=session_id).first()
        if not prediction:
            return jsonify({'success': False, 'message': 'Report not found'}), 404
        
        # Delete the prediction
        db.session.delete(prediction)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Report deleted successfully'})
    
    except Exception as e:
        db.session.rollback()
        print(f"Error in admin delete report: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/download_medical_report/<int:report_id>')
@login_required
@admin_required
def admin_download_medical_report(report_id):
    """Admin download of a medical report file"""
    try:
        # Get the medical report
        report = UserMedicalReport.query.get(report_id)
        if not report:
            flash('Medical report not found.', 'error')
            return redirect(url_for('admin_users'))
        
        # Get user info
        user = User.query.get(report.user_id)
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('admin_users'))
        
        # Check if file exists
        file_path = os.path.join('uploads', report.filename)
        if not os.path.exists(file_path):
            flash('Report file not found.', 'error')
            return redirect(url_for('admin_users'))
        
        # Send the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"Medical_Report_{user.username}_{report.original_filename}"
        )
    
    except Exception as e:
        print(f"Error in admin download medical report: {str(e)}")
        flash(f'Error downloading medical report: {str(e)}', 'error')
        return redirect(url_for('admin_users'))

# --- NLP-based Symptom Analysis ---
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except Exception as e:
    print(f"spaCy not available: {e}")
    SPACY_AVAILABLE = False

def analyze_symptoms_nlp(symptoms_text):
    if not SPACY_AVAILABLE:
        return None  # Fallback to old method
    doc = nlp(symptoms_text.lower())
    found = set()
    for ent in doc.ents:
        if ent.label_ in ['SYMPTOM', 'DISEASE', 'CONDITION']:
            found.add(ent.text)
    # Simple token/lemma-based rules
    for token in doc:
        if token.lemma_ in ['pain', 'hurt', 'ache', 'pressure']:
            for child in token.children:
                if child.lemma_ in ['chest', 'arm', 'jaw', 'back']:
                    found.add('chest pain')
        if token.lemma_ in ['faint', 'collapse', 'unconscious']:
            found.add('fainting')
        if token.lemma_ in ['breath', 'breathe'] and token.head.lemma_ in ['short', 'difficulty', 'hard']:
            found.add('shortness of breath')
        if token.lemma_ in ['sweat', 'sweating']:
            found.add('sweating')
        if token.lemma_ in ['dizzy', 'dizziness', 'lightheaded']:
            found.add('dizziness')
    # --- Added: Direct phrase matching for robustness ---
    text = symptoms_text.lower()
    phrase_symptoms = [
        'chest pain', 'shortness of breath', 'fainting', 'dizziness', 'palpitations',
        'fatigue', 'weakness', 'swelling', 'irregular heartbeat', 'lightheadedness',
        'nausea', 'sweating', 'jaw pain', 'pain radiating', 'pain spreading to arm',
        'chest discomfort', 'chest tightness', 'difficulty breathing', 'cold sweat',
        'nausea with chest pain', 'dizziness with chest pain', 'severe chest pain',
        'crushing chest pain', 'intense chest pressure', 'unable to breathe',
        "can't breathe", 'gasping for air', 'choking', 'heart attack', 'heart stopped',
        'lost consciousness'
    ]
    for phrase in phrase_symptoms:
        if phrase in text:
            found.add(phrase)
    # Classify
    if 'chest pain' in found or 'fainting' in found or 'shortness of breath' in found:
        return {
            'is_emergency': True,
            'risk_level': 'high',
            'concerns': list(found),
            'emergency_reason': 'Detected serious symptom(s): ' + ', '.join(found) if found else 'Possible cardiac emergency detected.'
        }
    if 'dizziness' in found or 'sweating' in found:
        return {'is_emergency': False, 'risk_level': 'medium', 'concerns': list(found)}
    if found:
        return {'is_emergency': False, 'risk_level': 'low', 'concerns': list(found)}
    return {'is_emergency': False, 'risk_level': 'low', 'concerns': []}

# --- Use NLP in chat assessment ---
def analyze_symptoms(symptoms_text):
    nlp_result = analyze_symptoms_nlp(symptoms_text)
    if nlp_result is not None:
        return nlp_result
    # Fallback to old method if spaCy not available
    analysis = {
        'is_emergency': False,
        'emergency_reason': '',
        'risk_level': 'low',
        'concerns': [],
        'recommendations': []
    }
    text = symptoms_text.lower()
    emergency_indicators = [
        'severe chest pain', 'crushing chest pain', 'intense chest pressure',
        'unable to breathe', "can't breathe", 'gasping for air',
        'choking', 'heart attack', 'heart stopped',
        'fainting', 'fainted', 'lost consciousness'
    ]
    high_risk_indicators = [
        'chest pain', 'chest discomfort', 'chest tightness',
        'shortness of breath', 'difficulty breathing', 
        'pain radiating', 'pain spreading to arm', 'jaw pain',
        'nausea with chest pain', 'cold sweat', 'dizziness with chest pain'
    ]
    moderate_risk_indicators = [
        'fatigue', 'weakness', 'swelling', 'palpitations',
        'irregular heartbeat', 'lightheadedness'
    ]
    for indicator in emergency_indicators:
        if indicator in text:
            analysis['is_emergency'] = True
            analysis['emergency_reason'] = f"Symptoms indicate a possible cardiac emergency: {indicator}"
            return analysis
    for indicator in high_risk_indicators:
        if indicator in text:
            analysis['risk_level'] = 'high'
            analysis['concerns'].append(indicator)
            return analysis
    for indicator in moderate_risk_indicators:
        if indicator in text:
            analysis['risk_level'] = 'medium'
            analysis['concerns'].append(indicator)
            return analysis
    return analysis

@app.route('/api/chat_history', methods=['GET'])
@login_required
@patient_required
def api_chat_history():
    try:
        # Get all chat assessments for the current user
        user_id = current_user.id
        # Assuming Prediction model is used for chat assessments
        chat_assessments = Prediction.query.filter_by(source='chat_assessment').order_by(Prediction.timestamp.desc()).all()
        # If you have a UserPrediction model, filter by user_id as well
        # chat_assessments = UserPrediction.query.filter_by(user_id=user_id, source='chat_assessment').order_by(UserPrediction.timestamp.desc()).all()
        history = []
        for pred in chat_assessments:
            explanation = pred.explanation
            if isinstance(explanation, str):
                import json
                try:
                    explanation = json.loads(explanation)
                except Exception:
                    explanation = {}
            history.append({
                'timestamp': pred.timestamp.strftime('%Y-%m-%d %H:%M'),
                'risk_level': pred.risk_level,
                'probability': pred.probability,
                'detected_symptoms': explanation.get('concerns', []),
                'summary': explanation.get('risk_description', ''),
                'session_id': pred.session_id
            })
        return {'success': True, 'history': history}
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return {'success': False, 'error': 'Could not fetch chat history.'}, 500

@app.route('/test_feature_importance')
def test_feature_importance():
    """Test route to verify feature importance display"""
    from src.explainable_ai import get_feature_importance, create_feature_importance_plot, convert_plot_to_base64
    
    try:
        # Generate feature importance
        feature_importance = get_feature_importance('models')
        plot = create_feature_importance_plot(feature_importance)
        base64_img = convert_plot_to_base64(plot)
        
        # Create test explanation
        test_explanation = {
            'feature_importance_img': base64_img,
            'risk_level': 'Medium',
            'probability': 0.5,
            'risk_description': 'Test description'
        }
        
        return render_template('explanation.html',
            explanation=test_explanation,
            session_id='test',
            knn_probability=50.0,
            rf_probability=50.0,
            xgb_probability=50.0,
            knn_risk='Medium',
            rf_risk='Medium',
            xgb_risk='Medium',
        )
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/boxplot/<session_id>/<feature>')
def serve_boxplot(session_id, feature):
    """Serve boxplot images directly as PNG files instead of data URIs"""
    try:
        # Load the explanation data
        explanation_file = os.path.join('temp', f'prediction_{session_id}.pkl')
        if not os.path.exists(explanation_file):
            abort(404)
            
        with open(explanation_file, 'rb') as f:
            explanation = pickle.load(f)
        
        # Check if explanation has the 'explanation' nested structure
        if 'explanation' in explanation:
            actual_explanation = explanation['explanation']
        else:
            actual_explanation = explanation
        
        # Check if boxplot exists for this feature
        if 'boxplots' not in actual_explanation or feature not in actual_explanation['boxplots']:
            abort(404)
            
        # Get the base64 data and decode it
        base64_data = actual_explanation['boxplots'][feature]
        if not base64_data:
            abort(404)
            
        # Decode base64 to binary
        img_data = base64.b64decode(base64_data)
        
        # Return as PNG image
        return Response(
            img_data,
            mimetype='image/png',
            headers={
                'Content-Type': 'image/png',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        print(f"Error serving boxplot {feature} for session {session_id}: {str(e)}")
        abort(404)



if __name__ == '__main__':
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 8080))
    
    print("🚀 Starting Heart Care application...")
    print(f"🌐 Application will be available on port: {port}")
    print("📚 API Documentation: /help")
    
    # Initialize database immediately before starting the app
    print("🔧 Initializing database...")
    try:
        with app.app_context():
            init_db()
            print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Start the application
    app.run(
        host='0.0.0.0',  # Bind to all interfaces
        port=port,        # Use the port from environment
        debug=False,      # Disable debug mode for production
        threaded=True     # Enable threading for better performance
    )