from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='patient')  # 'admin', 'patient'
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    
    # Relationship with predictions
    predictions = db.relationship('UserPrediction', backref='user', lazy=True, cascade='all, delete-orphan')
    medical_reports = db.relationship('UserMedicalReport', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    

    
    def is_patient(self):
        """Check if user is patient"""
        return self.role == 'patient'
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'email_verified': self.email_verified
        }

class UserPrediction(db.Model):
    __tablename__ = 'user_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    input_data = db.Column(db.Text, nullable=False)  # JSON string
    prediction = db.Column(db.Text, nullable=False)  # JSON string
    individual_predictions = db.Column(db.Text)  # JSON string
    explanation = db.Column(db.Text)  # JSON string
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    source = db.Column(db.String(50), default='manual_entry')
    risk_level = db.Column(db.String(20))
    probability = db.Column(db.Float)
    notes = db.Column(db.Text)  # User notes
    is_shared = db.Column(db.Boolean, default=False)  # Share with doctor
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'input_data': json.loads(self.input_data) if self.input_data else {},
            'prediction': json.loads(self.prediction) if self.prediction else {},
            'individual_predictions': json.loads(self.individual_predictions) if self.individual_predictions else {},
            'explanation': json.loads(self.explanation) if self.explanation else {},
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'risk_level': self.risk_level,
            'probability': self.probability,
            'notes': self.notes,
            'is_shared': self.is_shared
        }

class UserMedicalReport(db.Model):
    __tablename__ = 'user_medical_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    content = db.Column(db.Text)  # Extracted text content
    analysis_results = db.Column(db.Text)  # JSON string of analysis results
    is_processed = db.Column(db.Boolean, default=False)
    file_size = db.Column(db.Integer)  # File size in bytes
    
    def to_dict(self):
        """Convert report to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_type': self.file_type,
            'upload_date': self.upload_date.isoformat(),
            'content': self.content,
            'analysis_results': json.loads(self.analysis_results) if self.analysis_results else {},
            'is_processed': self.is_processed,
            'file_size': self.file_size
        }

class ConversationHistory(db.Model):
    __tablename__ = 'conversation_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Optional for guest users
    session_id = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)  # True for user, False for bot
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    state = db.Column(db.String(50))  # Conversation state
    
    def to_dict(self):
        """Convert conversation to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'message': self.message,
            'is_user': self.is_user,
            'timestamp': self.timestamp.isoformat(),
            'state': self.state
        } 