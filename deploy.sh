#!/bin/bash

# Heart Disease Predictor - Render Deployment Script
echo "🚀 Starting deployment preparation..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project root."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found."
    exit 1
fi

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo "❌ Error: render.yaml not found."
    exit 1
fi

echo "✅ All required files found!"

# Create necessary directories if they don't exist
mkdir -p uploads
mkdir -p temp
mkdir -p models
mkdir -p data

echo "📁 Directories created/verified"

# Check if models exist, if not provide instructions
if [ ! -f "models/heart_disease_model.pkl" ]; then
    echo "⚠️  Warning: ML models not found in models/ directory"
    echo "   The application will train models on first run (may take time)"
    echo "   For faster deployment, consider pre-training models locally"
fi

# Check for Tesseract dependency
echo "🔍 Checking Tesseract dependency..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract is installed locally"
    tesseract --version | head -n 1
else
    echo "⚠️  Tesseract not found locally (will be installed on Render)"
    echo "   This is normal for local development"
fi

# Check for Poppler dependency
echo "🔍 Checking Poppler dependency..."
if command -v pdftoppm &> /dev/null; then
    echo "✅ Poppler is installed locally"
else
    echo "⚠️  Poppler not found locally (will be installed on Render)"
    echo "   This is normal for local development"
fi

echo "🎯 Deployment preparation complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Push your code to GitHub"
echo "2. Connect your repository to Render"
echo "3. Set environment variables in Render dashboard"
echo "4. Deploy!"
echo ""
echo "🔧 Required Environment Variables:"
echo "   - SECRET_KEY (generate with: python -c 'import secrets; print(secrets.token_hex(32))')"
echo "   - GEMINI_API_KEY (optional but recommended)"
echo "   - MAIL_USERNAME (optional)"
echo "   - MAIL_PASSWORD (optional)"
echo "   - MAIL_DEFAULT_SENDER (optional)"
echo ""
echo "🔍 Tesseract & Poppler:"
echo "   - Will be automatically installed during Render build"
echo "   - No additional configuration needed"
echo ""
echo "🌐 Your app will be available at: https://your-app-name.onrender.com"
echo ""
echo "⚠️  Important Notes:"
echo "   - Initial deployment may take 5-10 minutes due to ML libraries"
echo "   - Tesseract OCR will be available for PDF/image processing"
echo "   - File uploads work but won't persist (Render limitation)" 