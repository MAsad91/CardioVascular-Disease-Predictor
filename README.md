
# ❤️ Heart Disease Prediction Platform

A **comprehensive web platform** designed to **predict heart disease risk** using advanced **machine learning models**, **explainable AI**, and a suite of interactive tools. Built for **healthcare professionals**, **researchers**, and **individuals** seeking reliable insights into cardiovascular health.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit%20Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

### 🔍 **Multi-Model Prediction**
- **K-Nearest Neighbors (KNN)** - Fast and accurate predictions
- **Random Forest** - Robust ensemble learning
- **XGBoost** - High-performance gradient boosting
- **Support Vector Machine (SVM)** - Advanced classification
- **Logistic Regression** - Interpretable baseline model
- **Decision Tree** - Transparent decision paths

### ⚡ **Quick Risk Assessment**
- **Lightweight Model** - Get predictions in seconds
- **Minimal Input Required** - Essential parameters only
- **Real-time Results** - Instant feedback

### 🤖 **Explainable AI**
- **Feature Importance** - Understand which factors matter most
- **SHAP Values** - Detailed prediction explanations
- **Model Interpretability** - Clear reasoning behind predictions
- **Visual Explanations** - Charts and graphs for insights

### 📄 **Document Analysis**
- **PDF Upload Support** - Analyze medical reports and ECGs
- **Image Processing** - Extract data from scanned documents
- **OCR Integration** - Text extraction from images
- **Report Generation** - Automated PDF reports

### 📊 **Admin Dashboard**
- **User Management** - Track registered users
- **Report Analytics** - View prediction statistics
- **System Monitoring** - Performance metrics
- **Data Export** - Download user data

### 💬 **Interactive Chatbot**
- **Health Tips** - Personalized lifestyle advice
- **Prediction Explanations** - Detailed model insights
- **Risk Factors** - Educational content
- **24/7 Support** - Always available assistance

### 🔐 **Security Features**
- **User Authentication** - Secure login system
- **Password Reset** - Email-based recovery
- **Session Management** - Secure user sessions
- **Data Privacy** - GDPR compliant

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/MAsad91/CardioVascular-Disease-Predictor.git
cd CardioVascular-Disease-Predictor
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
DATABASE_URL=sqlite:///instance/heart_disease.db
```

### 5. Initialize Database
```bash
python -c "from app import create_app; app = create_app(); app.app_context().push(); from models import db; db.create_all()"
```

### 6. Train Models (Optional)
```bash
# Train all models
python src/train_model.py

# Or train specific models
python src/multi_model.py
```

### 7. Launch the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

---

## 📁 Project Structure

```
CardioVascular-Disease-Predictor/
├── app.py                    # Main Flask application
├── config.py                 # Application configuration
├── models.py                 # Database models
├── requirements.txt          # Python dependencies
├── requirements.md           # Detailed requirements
├── .gitignore               # Git ignore rules
├── .env                     # Environment variables (create this)
│
├── data/                    # Datasets
│   └── heart_disease.csv    # Sample heart disease dataset
│
├── src/                     # Source code
│   ├── chatbot.py           # Chatbot implementation
│   ├── download_data.py     # Data download utilities
│   ├── ecg_analyzer.py      # ECG analysis tools
│   ├── explainable_ai.py    # AI explanation module
│   ├── gemini_chatbot.py    # Gemini AI integration
│   ├── heart_disease_info.py # Disease information
│   ├── heart_disease_predictor.py # Main predictor
│   ├── image_processor_new.py # Image processing
│   ├── multi_model.py       # Multi-model training
│   ├── pdf_report.py        # PDF generation
│   ├── predict.py           # Prediction utilities
│   ├── preprocess.py        # Data preprocessing
│   └── train_model.py       # Model training
│
├── static/                  # Static assets
│   └── css/
│       └── style.css        # Custom styles
│
├── templates/               # HTML templates
│   ├── admin/              # Admin panel templates
│   ├── about.html          # About page
│   ├── assessment.html     # Assessment form
│   ├── chat.html          # Chat interface
│   ├── index.html         # Homepage
│   ├── login.html         # Login page
│   ├── profile.html       # User profile
│   ├── result.html        # Results display
│   ├── signup.html        # Registration
│   └── upload.html        # File upload
│
├── instance/               # (Auto-generated) Database files
├── models/                 # (Auto-generated) Trained models
├── temp/                   # (Auto-generated) Temporary files
├── uploads/                # (Auto-generated) User uploads
└── venv/                   # (Auto-generated) Virtual environment
```

---

## 🚀 Usage Guide

### For Users

1. **Register/Login** - Create an account or sign in
2. **Quick Assessment** - Use the fast prediction tool
3. **Detailed Assessment** - Complete comprehensive health questionnaire
4. **Upload Documents** - Analyze medical reports and ECGs
5. **View Results** - Get detailed predictions with explanations
6. **Chat with AI** - Ask questions and get health tips
7. **Download Reports** - Save results as PDF

### For Administrators

1. **Access Admin Panel** - Navigate to `/admin`
2. **User Management** - View and manage user accounts
3. **Report Analytics** - Monitor system usage
4. **Data Export** - Download user data and reports

### For Developers

1. **Model Training** - Use scripts in `src/` directory
2. **API Integration** - Extend with REST API endpoints
3. **Custom Models** - Add new ML algorithms
4. **UI Customization** - Modify templates and styles

---

## 🔧 Configuration

### Environment Variables
```env
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
DATABASE_URL=sqlite:///instance/heart_disease.db
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
```

### Model Configuration
Edit `src/heart_disease_predictor.py` to customize:
- Feature selection
- Model parameters
- Prediction thresholds
- Explanation methods

---

## 📊 API Endpoints

### Authentication
- `POST /signup` - User registration
- `POST /login` - User login
- `POST /logout` - User logout
- `POST /forgot_password` - Password reset

### Predictions
- `POST /quick_assessment` - Quick risk assessment
- `POST /assessment` - Detailed assessment
- `POST /upload` - File upload and analysis
- `GET /result/<id>` - Get prediction results

### Admin
- `GET /admin/users` - List all users
- `GET /admin/reports` - View all reports
- `POST /admin/export` - Export data

---

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest tests/
```

### Test Coverage
```bash
# Install coverage
pip install coverage

# Run with coverage
coverage run -m pytest
coverage report
```

---

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Heroku Deployment
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure all tests pass

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

### Get Help
- 📧 **Email**: [asadnisar108@gmail.com]
- 🐛 **Issues**: [GitHub Issues](https://github.com/MAsad91/CardioVascular-Disease-Predictor/issues)
- 📖 **Documentation**: [Wiki](https://github.com/MAsad91/CardioVascular-Disease-Predictor/wiki)

### Demo Requests
For access to demo data, models, or live demonstrations, please contact us via email.

---

## 🙏 Acknowledgments

- **Scikit-learn** - Machine learning algorithms
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **SHAP** - Model explainability

---

### 🚀 **Let's fight heart disease with the power of AI!**

*Built with ❤️ for better healthcare outcomes*
