# Testing Guide

## Local Testing Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
python -m pip install Flask==2.3.3 Flask-SQLAlchemy==3.0.5 Flask-Login Flask-Mail python-dotenv
python -m pip install pandas numpy scikit-learn matplotlib opencv-python
python -m pip install xgboost fpdf2 pytesseract pdf2image PyMuPDF reportlab seaborn
```

### 2. Test Signup Functionality
```bash
# Test basic signup functionality
python -c "
import os
os.environ['FLASK_ENV'] = 'testing'
os.environ['DATABASE_URL'] = 'sqlite:///test_heart_disease.db'
from app import app, db, User
from datetime import datetime, timezone

with app.app_context():
    db.create_all()
    test_user = User(
        username='testuser',
        email='test@example.com',
        created_at=datetime.now(timezone.utc),
        is_active=True,
        email_verified=False
    )
    test_user.set_password('testpassword123')
    db.session.add(test_user)
    db.session.commit()
    print('✅ Signup test passed!')
"
```

### 3. Test Flask Routes
```bash
# Test signup route with Flask test client
python -c "
import os
os.environ['FLASK_ENV'] = 'testing'
os.environ['DATABASE_URL'] = 'sqlite:///test_heart_disease.db'
from app import app, db, User

with app.test_client() as client:
    with app.app_context():
        db.create_all()
        response = client.post('/signup', data={
            'user_name': 'testuser123',
            'email': 'test123@example.com',
            'password': 'testpassword123',
            'confirm_password': 'testpassword123'
        }, follow_redirects=True)
        print(f'Status: {response.status_code}')
        user = User.query.filter_by(username='testuser123').first()
        if user:
            print('✅ Route test passed!')
            db.session.delete(user)
            db.session.commit()
"
```

## Key Testing Points

### ✅ What to Test Before Deployment:
1. **Database Operations**: User creation, password hashing, datetime handling
2. **Route Functionality**: Signup, login, logout routes
3. **Error Handling**: Invalid inputs, duplicate users, database errors
4. **Performance**: Response times, database query optimization
5. **Cross-version Compatibility**: Python version compatibility (especially datetime.UTC vs timezone.utc)

### 🔧 Common Issues Fixed:
- **datetime.UTC**: Not available in Python < 3.11, use `timezone.utc` instead
- **Database Connections**: Use connection pooling for better performance
- **Import Errors**: Ensure all dependencies are installed locally

### 📋 Testing Checklist:
- [ ] App imports without errors
- [ ] Database tables can be created
- [ ] User creation works
- [ ] Password hashing works
- [ ] Signup route responds correctly
- [ ] No deprecation warnings
- [ ] Performance is acceptable

## Best Practices

1. **Always test locally first** before pushing to production
2. **Use virtual environments** to isolate dependencies
3. **Test with different Python versions** if possible
4. **Monitor logs** for any errors or warnings
5. **Clean up test data** after testing
6. **Document any issues** found during testing

## Troubleshooting

### Common Errors:
- `ModuleNotFoundError`: Install missing dependencies
- `datetime.UTC` errors: Use `timezone.utc` instead
- Database connection issues: Check connection string and pooling settings
- Import errors: Ensure all required packages are installed

### Performance Issues:
- Check database query optimization
- Monitor connection pooling settings
- Look for unnecessary database operations
- Check for memory leaks in long-running processes 