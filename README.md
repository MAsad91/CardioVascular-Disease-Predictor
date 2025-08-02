
# Heart Disease Prediction System

A comprehensive Flask-based web application for heart disease prediction using machine learning models.

## Features

- **Multi-Model Prediction**: Uses KNN, Random Forest, and XGBoost models
- **Explainable AI**: Provides detailed explanations of predictions
- **User Authentication**: Secure login/signup system
- **File Upload**: PDF medical report processing
- **Chat Assessment**: AI-powered conversational assessment
- **Admin Dashboard**: User and report management
- **PDF Reports**: Downloadable assessment reports

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Heart-Disease
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create `.env` file):
```env
SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-email-password
MAIL_DEFAULT_SENDER=your-email@gmail.com
```

4. Run the application:
```bash
python app.py
```

5. Visit `http://localhost:8080`

## Deployment on Render

### Prerequisites
- Render account
- GitHub repository with your code

### Deployment Steps

1. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" → "Web Service"

2. **Configure the Service**:
   - **Name**: `heart-disease-predictor`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

3. **Set Environment Variables**:
   - `SECRET_KEY`: Generate a secure random key
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `MAIL_USERNAME`: Your email for notifications
   - `MAIL_PASSWORD`: Your email password/app password
   - `MAIL_DEFAULT_SENDER`: Your email address
   - `FLASK_ENV`: `production`

4. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Environment Variables for Render

```env
SECRET_KEY=your-secure-secret-key
GEMINI_API_KEY=your-gemini-api-key
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=your-email@gmail.com
FLASK_ENV=production
```

### Database Configuration

The application uses SQLite by default, which works on Render. For production, consider:

1. **PostgreSQL** (Recommended):
   - Add PostgreSQL service in Render
   - Update `DATABASE_URL` environment variable
   - Install `psycopg2-binary` in requirements.txt

2. **MongoDB Atlas**:
   - Set up MongoDB Atlas cluster
   - Update `MONGODB_URI` environment variable

### File Storage

For production file uploads, consider:

1. **AWS S3**:
   - Set up S3 bucket
   - Install `boto3` in requirements.txt
   - Update upload logic

2. **Cloudinary**:
   - Set up Cloudinary account
   - Install `cloudinary` in requirements.txt
   - Update image processing

## Project Structure

```
Heart-Disease/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── models.py             # Database models
├── requirements.txt      # Python dependencies
├── render.yaml          # Render configuration
├── Procfile             # Process file for deployment
├── runtime.txt          # Python version specification
├── data/                # Dataset files
├── models/              # Trained ML models
├── src/                 # Source code modules
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
└── uploads/             # File upload directory
```

## API Endpoints

- `GET /` - Main dashboard
- `POST /api/predict` - Heart disease prediction
- `GET /reports` - User prediction history
- `POST /upload` - File upload endpoint
- `GET /chat` - Chat assessment

## Machine Learning Models

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **XGBoost**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue on GitHub or contact the development team.
