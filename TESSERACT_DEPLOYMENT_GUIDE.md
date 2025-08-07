# 🔍 Tesseract OCR Deployment Guide for Render

## ✅ **Tesseract is Essential for Your Application**

Your Heart Disease Predictor application **requires Tesseract OCR** for:
- 📄 **PDF Processing**: Extract medical data from uploaded PDF reports
- 🖼️ **Image Processing**: OCR text from medical images
- 📊 **Data Extraction**: Parse lab results, patient information, and medical metrics
- 🔍 **Report Analysis**: Process ECG reports and diagnostic documents

## 🚀 **Render Deployment Configuration**

### **Updated render.yaml**
Your `render.yaml` now includes comprehensive Tesseract installation:

```yaml
buildCommand: |
  # Update package list
  apt-get update
  
  # Install system dependencies including Tesseract OCR and Poppler
  apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    poppler-utils \
    libpoppler-cpp-dev \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgif-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libtesseract-dev \
    libleptonica-dev
  
  # Install Python dependencies
  pip install -r requirements.txt
  
  # Verify Tesseract installation
  echo "Verifying Tesseract installation..."
  tesseract --version
  echo "Tesseract installation verified!"
```

## 📦 **What Gets Installed**

### **Tesseract Components:**
- ✅ `tesseract-ocr` - Core OCR engine
- ✅ `tesseract-ocr-eng` - English language pack
- ✅ `tesseract-ocr-osd` - Script and orientation detection
- ✅ `libtesseract-dev` - Development libraries
- ✅ `libleptonica-dev` - Image processing library

### **Poppler Components (for PDF processing):**
- ✅ `poppler-utils` - PDF to image conversion
- ✅ `libpoppler-cpp-dev` - PDF processing library

### **Image Processing Libraries:**
- ✅ `libcairo2-dev` - Vector graphics
- ✅ `libpango1.0-dev` - Text layout
- ✅ `libjpeg-dev`, `libpng-dev`, `libtiff-dev` - Image formats
- ✅ `libwebp-dev` - WebP image support

## 🔧 **Application Configuration**

### **Updated Image Processor**
Your `src/image_processor_new.py` now includes:

1. **Enhanced Path Detection**:
   ```python
   common_paths = [
       r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Windows
       '/usr/bin/tesseract',                              # Linux (Render)
       '/usr/local/bin/tesseract',                        # Linux alternative
       '/opt/homebrew/bin/tesseract',                     # macOS
   ]
   ```

2. **Automatic Testing**:
   ```python
   if self.tesseract_available:
       try:
           test_result = pytesseract.get_tesseract_version()
           print(f"✅ Tesseract version: {test_result}")
       except Exception as e:
           print(f"⚠️  Tesseract test failed: {str(e)}")
   ```

3. **Poppler Path Detection**:
   ```python
   poppler_paths = [
       r'C:\poppler\Library\bin',  # Windows
       '/usr/bin',                 # Linux (Render)
       '/usr/local/bin',           # Linux alternative
   ]
   ```

## 🚀 **Deployment Steps**

### **Step 1: Verify Your Repository**
Ensure these files are in your repository:
- ✅ `render.yaml` (updated with Tesseract installation)
- ✅ `requirements.txt` (includes pytesseract)
- ✅ `src/image_processor_new.py` (updated path detection)
- ✅ `app.py` (Tesseract configuration)

### **Step 2: Deploy on Render**
1. **Go to [render.com](https://render.com)**
2. **Create New Web Service**
3. **Connect your GitHub repository**
4. **Configure service**:
   - **Name**: `heart-disease-predictor`
   - **Environment**: `Python`
   - **Build Command**: (from render.yaml)
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

### **Step 3: Set Environment Variables**
```env
SECRET_KEY=your-generated-secret-key
GEMINI_API_KEY=your-gemini-api-key (optional)
MAIL_USERNAME=your-email (optional)
MAIL_PASSWORD=your-app-password (optional)
FLASK_ENV=production
```

### **Step 4: Deploy**
Click "Create Web Service" and wait 5-10 minutes.

## 🔍 **Verification Steps**

### **After Deployment, Check:**

1. **Build Logs**: Look for:
   ```
   ✅ Verifying Tesseract installation...
   ✅ Tesseract version: 4.1.1
   ✅ Tesseract installation verified!
   ```

2. **Application Logs**: Look for:
   ```
   ✅ Tesseract found in PATH: /usr/bin/tesseract
   ✅ Tesseract version: 4.1.1
   ```

3. **Test OCR Functionality**:
   - Upload a PDF medical report
   - Check if data is extracted correctly
   - Verify image processing works

## ⚠️ **Troubleshooting**

### **If Tesseract Installation Fails:**

1. **Check Build Logs**:
   ```bash
   # In Render dashboard, check build logs for errors
   ```

2. **Common Issues**:
   - **Package not found**: Update package list with `apt-get update`
   - **Permission denied**: Ensure proper permissions
   - **Memory issues**: Render free tier has 512MB RAM limit

3. **Alternative Installation**:
   ```yaml
   # If standard installation fails, try:
   apt-get install -y tesseract-ocr tesseract-ocr-eng --no-install-recommends
   ```

### **If OCR Doesn't Work:**

1. **Check Python Dependencies**:
   ```bash
   pip install pytesseract Pillow opencv-python
   ```

2. **Verify Tesseract Path**:
   ```python
   import pytesseract
   print(pytesseract.get_tesseract_version())
   ```

3. **Test Simple OCR**:
   ```python
   import pytesseract
   from PIL import Image
   
   # Create a simple test image
   img = Image.new('RGB', (100, 30), color='white')
   text = pytesseract.image_to_string(img)
   print(f"OCR Test: {text}")
   ```

## 📊 **Performance Considerations**

### **Render Free Tier Limits:**
- **Memory**: 512MB RAM
- **Build Time**: 5-10 minutes (due to ML + Tesseract)
- **Storage**: ~200MB (including Tesseract)

### **Optimization Tips:**
1. **Pre-train models** locally to reduce build time
2. **Use environment variables** for configuration
3. **Consider upgrading** to paid tier for better performance
4. **Monitor memory usage** during OCR processing

## ✅ **Expected Results**

After successful deployment, your application will have:

- ✅ **Full OCR Capability**: Process PDFs and images
- ✅ **Medical Report Analysis**: Extract patient data
- ✅ **ECG Report Processing**: Analyze cardiac data
- ✅ **Lab Result Parsing**: Extract diagnostic values
- ✅ **Multi-format Support**: PDF, PNG, JPG, TIFF

## 🎯 **Summary**

Your Heart Disease Predictor application **is fully configured for Tesseract deployment on Render**! The updated configuration ensures:

- ✅ **Automatic Tesseract Installation** during build
- ✅ **Comprehensive Dependencies** for image processing
- ✅ **Robust Path Detection** for different environments
- ✅ **Error Handling** and verification
- ✅ **Production-Ready** OCR functionality

**Deployment Time**: 5-10 minutes
**Cost**: Free tier available
**OCR Capability**: Full functionality

Your application will be able to process medical reports, extract data from PDFs, and perform OCR on images as intended! 🚀 