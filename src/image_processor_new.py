import os
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import pdf2image
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from .ecg_analyzer import ECGAnalyzer
from PIL import ImageEnhance
import traceback
import json
import fitz # PyMuPDF
import logging

class MedicalReportProcessor:
    """
    Class to process medical reports using OCR to extract patient health data.
    Supports both image and PDF formats.
    """
    
    def __init__(self):
        """Initialize the medical report processor."""
        self.tesseract_available = False
        try:
            import pytesseract
            
            # Try common Tesseract installation paths (including Render/Linux paths)
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
                '/opt/homebrew/bin/tesseract',  # macOS Homebrew
                '/usr/local/opt/tesseract/bin/tesseract'  # macOS Homebrew alternative
            ]
            
            # Check if tesseract is in PATH
            from shutil import which
            tesseract_path = which('tesseract')
            
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.tesseract_available = True
                print(f"✅ Tesseract found in PATH: {tesseract_path}")
            else:
                # Try common paths
                for path in common_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self.tesseract_available = True
                        print(f"✅ Tesseract found at: {path}")
                        break
                
                if not self.tesseract_available:
                    print("\n❌ Tesseract not found. Please follow these steps to install:")
                    print("1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki")
                    print("2. Install to default location (C:\\Program Files\\Tesseract-OCR)")
                    print("3. Add Tesseract to your PATH environment variable")
                    print("4. Restart your computer if needed.")
                    print("\nFor Render deployment, Tesseract will be installed automatically during build.")
        except ImportError:
            print("❌ Warning: pytesseract not installed. OCR functionality will be limited.")
            print("Please install pytesseract: pip install pytesseract")
        
        # Test Tesseract availability
        if self.tesseract_available:
            try:
                # Simple test to verify Tesseract is working
                test_result = pytesseract.get_tesseract_version()
                print(f"✅ Tesseract version: {test_result}")
            except Exception as e:
                print(f"⚠️  Tesseract found but test failed: {str(e)}")
                self.tesseract_available = False
    
    def is_pdf(self, file_path):
        """Check if the file is a PDF"""
        try:
            return file_path.lower().endswith('.pdf')
        except Exception as e:
            print(f"Error checking file type: {str(e)}")
            return False
    
    def convert_pdf_to_images(self, pdf_path):
        """Convert PDF pages to images using pdf2image"""
        try:
            print(f"Starting PDF conversion for: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                raise ValueError(f"PDF file not found: {pdf_path}")
            
            # Try to find poppler in common locations (including Linux paths for Render)
            poppler_paths = [
                r'C:\poppler\Library\bin',  # Windows
                '/usr/bin',  # Linux (Render)
                '/usr/local/bin',  # Linux alternative
                '/opt/homebrew/bin'  # macOS Homebrew
            ]
            
            poppler_path = None
            for path in poppler_paths:
                if os.path.exists(path):
                    poppler_path = path
                    break
            
            if not poppler_path:
                print("\n⚠️  Poppler not found in common locations. Trying system PATH...")
                # Try to find poppler in system PATH
                from shutil import which
                poppler_bin = which('pdftoppm')
                if poppler_bin:
                    poppler_path = os.path.dirname(poppler_bin)
                    print(f"✅ Found Poppler in PATH: {poppler_path}")
                else:
                    print("\n❌ Poppler not found. Please follow these steps to install:")
                    print("1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/")
                    print("2. Extract to C:\\poppler")
                    print("3. Add C:\\poppler\\Library\\bin to system PATH")
                    print("4. Restart your application")
                    print("\nFor Render deployment, Poppler will be installed automatically during build.")
                    raise ValueError("Poppler not found. Please install poppler-utils.")
            
            print(f"✅ Using Poppler path: {poppler_path}")
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, poppler_path=poppler_path)
            
            if not images:
                print("No images extracted from PDF")
                raise ValueError("Failed to extract any images from the PDF")
            
            print(f"Successfully converted PDF to {len(images)} images")
            return images
            
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            print(f"PDF conversion error: {str(e)}")
            raise ValueError(f"Error converting PDF to images: {str(e)}")
        except Exception as e:
            print(f"PDF conversion failed with error: {str(e)}")
            if "poppler" in str(e).lower():
                raise ValueError(
                    f"Poppler error using path '{poppler_path}'. Please ensure Poppler is properly installed "
                    "and the path is correct. You can download Poppler from: "
                    "https://github.com/oschwartz10612/poppler-windows/releases/"
                )
            raise ValueError(f"Failed to convert PDF: {str(e)}")
    
    def process_file(self, file_path):
        """Process a file (PDF or image) and extract data"""
        try:
            print(f"\nProcessing file: {file_path}")
            extracted_data = None
            
            if self.is_pdf(file_path):
                # Handle PDF
                print("Processing PDF file")
                images = self.convert_pdf_to_images(file_path)
                for i, img in enumerate(images):
                    # Convert PIL Image to numpy array
                    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    # Process the image
                    processed_img = self.preprocess_image(img_array)
                    text = self.extract_text(processed_img)
                    
                    # Detect report type
                    if self.detect_report_type(text) == 'ecg':
                        page_data = self.extract_ecg_data(text)
                    else:
                        page_data = self.extract_diagnostic_data(text)
                    
                    if page_data:
                        # Validate the extracted data
                        extracted_data = self.validate_extracted_data(page_data)
                        break  # Use first page with valid data
            else:
                # Handle image file
                print("Processing image file")
                img = cv2.imread(file_path)
                if img is not None:
                    processed_img = self.preprocess_image(img)
                    text = self.extract_text(processed_img)
                    
                    # Detect report type
                    if self.detect_report_type(text) == 'ecg':
                        page_data = self.extract_ecg_data(text)
                    else:
                        page_data = self.extract_diagnostic_data(text)
                    
                    if page_data:
                        # Validate the extracted data
                        extracted_data = self.validate_extracted_data(page_data)
                else:
                    print(f"Could not read image: {file_path}")
            
            return extracted_data
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_skew_angle(self, image):
        """Calculate skew angle of text in image"""
        try:
            # Find all non-zero points in the image
            coords = np.column_stack(np.where(image > 0))
            
            if len(coords) < 20:  # Not enough points to calculate angle
                return 0
            
            # Calculate orientation using PCA
            mean = np.mean(coords, axis=0)
            centered = coords - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Get the angle of the largest eigenvector
            angle = np.degrees(np.arctan2(eigvecs[-1, 1], eigvecs[-1, 0]))
            
            # Normalize angle to be between -45 and 45 degrees
            while angle < -45:
                angle += 90
            while angle > 45:
                angle -= 90
                
            return angle
            
        except Exception as e:
            print(f"Error calculating skew angle: {str(e)}")
            return 0
    
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR results.
        Optimized for both lab reports and ECG images.
        """
        try:
            print(f"Starting image preprocessing, input shape: {image.shape}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            print("Converted to grayscale")
            
            # Scale the image - larger scale for camera images
            scale_factor = 2.5  # Increased from 2.0 for better detail
            scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            print(f"Scaled image by factor of {scale_factor}")
            
            # Apply bilateral filter to remove noise while preserving edges
            denoised = cv2.bilateralFilter(scaled, 11, 85, 85)  # Increased values for better noise removal
            print("Applied denoising")
            
            # Enhance contrast using CLAHE with adjusted parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Increased clip limit
            contrasted = clahe.apply(denoised)
            print("Enhanced contrast")
            
            # Apply adaptive thresholding with adjusted parameters
            thresh = cv2.adaptiveThreshold(
                contrasted,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                25,  # Increased block size for better text detection
                15   # Increased C value
            )
            print("Applied adaptive thresholding")
            
            # Deskew image if needed
            angle = self.get_skew_angle(thresh)
            if abs(angle) > 0.5:
                rotated = self.rotate_image(thresh, angle)
            else:
                rotated = thresh
            
            # Remove small noise
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, kernel)
            
            print("Completed image preprocessing")
            return cleaned
            
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            return image
    
    def rotate_image(self, image, angle):
        """Rotate the image by the given angle"""
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new image dimensions
            abs_cos = abs(rotation_matrix[0,0])
            abs_sin = abs(rotation_matrix[0,1])
            
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)
            
            # Adjust rotation matrix
            rotation_matrix[0, 2] += new_width/2 - center[0]
            rotation_matrix[1, 2] += new_height/2 - center[1]
            
            # Perform rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            print(f"Error rotating image: {str(e)}")
            return image
    
    def extract_text(self, image):
        """
        Extract text from an image using OCR.
        Enhanced for both lab reports and ECG images.
        """
        try:
            # Convert PIL Image to OpenCV format if needed
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Use the already preprocessed image if it was preprocessed before
            if len(image.shape) == 2:  # Already grayscale/preprocessed
                processed_image = image
            else:
                processed_image = self.preprocess_image(image)

            # Define different OCR configurations to try
            configs = [
                # Try table-optimized mode first
                {'psm': 6, 'oem': 3, 'config': '--dpi 300'},  # Assume uniform block of text
                {'psm': 4, 'oem': 3, 'config': '--dpi 300'},  # Assume single column of text
                {'psm': 3, 'oem': 3, 'config': '--dpi 300'},  # Auto-detect layout
                {'psm': 11, 'oem': 3, 'config': '--dpi 300'}, # Raw line detection
                {'psm': 1, 'oem': 3, 'config': '--dpi 300'},  # Auto orientation
            ]

            # Try each configuration and keep the best result
            best_text = ""
            best_confidence = 0.0
            
            print("\nTrying different OCR modes:")
            for config in configs:
                try:
                    custom_config = f"--oem {config['oem']} --psm {config['psm']} {config['config']}"
                    text = pytesseract.image_to_string(processed_image, config=custom_config)
                    
                    # Get confidence scores for the text
                    data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
                    confidences = [float(conf) for conf in data['conf'] if conf != '-1']
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    print(f"PSM {config['psm']} confidence: {avg_confidence:.1f}%")
                    
                    # Check if this result is better
                    if avg_confidence > best_confidence and len(text.strip()) > 0:
                        # Additional check for table format
                        if config['psm'] in [4, 6] and ':' in text:
                            # Boost confidence for table-like text
                            avg_confidence += 5
                        best_confidence = avg_confidence
                        best_text = text
                
                except Exception as e:
                    print(f"Error with config PSM {config['psm']}: {str(e)}")
                    continue
            
            if not best_text.strip():
                print("Warning: No text extracted from image")
                return ""
            
            print(f"\nBest confidence achieved: {best_confidence:.1f}%")
            
            # Clean up the extracted text
            lines = []
            for line in best_text.split('\n'):
                # Skip empty lines or lines with just special characters
                if not line.strip() or all(c in '.-_:|' for c in line.strip()):
                    continue
                # Clean up common OCR errors
                line = re.sub(r'[^\x00-\x7F]+', '', line)  # Remove non-ASCII
                line = re.sub(r'\s+', ' ', line)  # Normalize whitespace
                line = line.strip()
                
                # Special handling for table formats
                if ':' in line:
                    # Keep the original spacing around the colon
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        line = f"{key}: {value}"
                
                if line:
                    lines.append(line)

            final_text = '\n'.join(lines)
            
            # Save the raw OCR text for debugging
            print("\n" + "="*80)
            print("RAW OCR TEXT FOR DEBUGGING:")
            print("="*80)
            print(final_text)
            print("="*80)
            
            return final_text
            
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            traceback.print_exc()
            return ""
    
    def _is_valid_value(self, field, value):
        """
        Validate if a value is within acceptable range for a given field
        """
        try:
            # Convert value to float first
            float_value = float(value)
            
            # Define acceptable ranges for each field
            ranges = {
                'age': (1, 120),  # Widened age range to catch edge cases
                'thalach': (60, 200),
                'oldpeak': (0, 6.2),
                'restecg': (0, 2),
                'glucose': (50, 500),
                'urea': (10, 200),
                'creatinine': (0.1, 15.0)
            }
            
            if field in ranges:
                min_val, max_val = ranges[field]
                
                # Special handling for age
                if field == 'age':
                    print(f"\nValidating age value: {value} (type: {type(value)})")
                    print(f"Converted to float: {float_value} (type: {type(float_value)})")
                    
                    # Handle OCR errors where numbers get merged (like 253 instead of 53)
                    if float_value > 200:  # Definitely an OCR error
                        corrected_value = float(str(int(float_value))[-2:])  # Take last 2 digits
                        print(f"Correcting likely OCR error: {float_value} -> {corrected_value}")
                        float_value = corrected_value
                    
                    # If age is too low (like 1.0 or 3.0), it might be a misread
                    if float_value < 20:
                        print(f"Warning: Age value {float_value} is too low, likely a misread")
                        return False
                    # If age is reasonable, accept it
                    if min_val <= float_value <= max_val:
                        print(f"Valid age found: {float_value} (within range {min_val}-{max_val})")
                        return True
                    print(f"Warning: Age {float_value} outside valid range ({min_val}-{max_val})")
                    return False
                
                # For other fields, just check the range
                return min_val <= float_value <= max_val
                
        except (ValueError, TypeError) as e:
            print(f"Error validating {field} value '{value}': {str(e)}")
            return False
            
        return True

    def validate_extracted_data(self, data):
        """
        Validate extracted medical data and prepare for form prefill.
        """
        if not data or not isinstance(data, dict):
            return None
            
        validated_data = {}
        
        # Validate each field
        for field, value in data.items():
            if value is not None and self._is_valid_value(field, value):
                validated_data[field] = value
            else:
                print(f"Warning: Invalid value for {field}: {value}")
                
        return validated_data if validated_data else None

    def extract_diagnostic_data(self, text):
        """Extract data from diagnostic report format."""
        data = {}
        text_lower = text.lower()
        
        print("\nRaw text before processing:")
        print("---START OF TEXT---")
        print(text)
        print("---END OF TEXT---\n")
        
        # First extract basic patient info with more flexible patterns
        print("\nExtracting basic patient information:")
        basic_patterns = {
            'age': [
                # Handle OCR errors where numbers get merged
                r'age\s*[:]?\s*(?:2)?(\d{2})\s*(?:\(years?\)|years?|yrs?)',  # Matches "253 (Years)" -> captures "53"
                r'age\s*[-=:.]?\s*(?:2)?(\d{2})\s*(?:\(years?\)|years?|yrs?)',  # Matches "age: 253 years" -> captures "53"
                
                # Table cell formats
                r'(?:^|\n)[^\n:]*?(?:2)?(\d{2})\s*(?:years?|yrs?)?(?:\n|$)',  # Number followed by years in its own line/cell
                r'(?:^|\n)\s*(?:2)?(\d{2})\s*(?:years?|yrs?)?(?:\n|$)',  # Clean number in its own line/cell
                r'(?:^|\n)[^:]*?:\s*(?:2)?(\d{2})\s*(?:years?|yrs?)?(?:\n|$)',  # After any label with colon
                
                # Most common formats
                r'age\s*[:]?\s*(\d{1,3})',  # Simple "age: 53" or "age 53"
                r'age\s*[-=:.]?\s*(\d{1,3})',  # Flexible separator
                r'age\s*[^a-zA-Z\d]*(\d{1,3})',  # Any non-alphanumeric separator
                
                # Table formats
                r'(?:^|\n|\t)\s*age\s*[-=:.]?\s*(\d{1,3})',  # Age at start of line
                r'(?:^|\n|\t)\s*:\s*(\d{1,3})\s*(?:years?|yrs?)?',  # Just ": 53" in table
                
                # With units
                r'age\s*[:]?\s*(\d{1,3})\s*(?:years?|yrs?)',  # "age: 53 years"
                r'age\s*[-=:.]?\s*(\d{1,3})\s*(?:years?|yrs?)',  # "age - 53 years"
                
                # Special formats
                r'patient\s+age\s*[^a-zA-Z\d]*(\d{1,3})',  # "Patient Age: 53"
                r'age\s*(?:in\s+years?)?\s*[^a-zA-Z\d]*(\d{1,3})',  # "age in years: 53"
                r'years?\s*[:]?\s*(\d{1,3})',  # Just "Years: 53"
                
                # Backup patterns
                r'(?:^|\n|\s)(\d{1,3})\s*(?:years?|yrs?)\s*(?:old)?',  # "53 years old"
                r'(?:^|\n|\s)(?:age|years)[^a-zA-Z\d]*?(\d{1,3})'  # Fallback pattern
            ],
            'sex': [
                r'(?:gender|sex)\s*[:]\s*(male|female|m|f)\b',
                r'(?:gender|sex)\s*[-=]\s*(male|female|m|f)\b',
                r'(?:gender|sex)\s*[:]?\s*(male|female|m|f)\b',
                r'(?:^|\n|\s)(?:gender|sex)\s*[:]\s*(male|female|m|f)',
                r'gender\s*[^a-zA-Z]*(male|female|m|f)',
                r'\n\s*gender\s*:\s*(female|male)',  # Matches "Gender : Female"
                r'^\s*:\s*(female|male)\s*$',  # Matches ": Female" in a line
                r'(?:^|\n|\s)(?:gender|sex)[^a-zA-Z\d]*(male|female|m|f)',  # More flexible match
                r'patient\s+(?:gender|sex)\s*[^a-zA-Z\d]*(male|female|m|f)'  # Match "Patient Gender: Female"
            ]
        }

        # Enhanced test result patterns for tabular format
        test_patterns = {
            'glucose': [
                r'(?:glucose|sugar)\s*(?:\(random\))?\s*(?:[-:=]|\s+)\s*(\d{2,3})',
                r'serum\s+glucose\s*(?:\(random\))?\s*(?:[-:=]|\s+)\s*(\d{2,3})',
                r'(?:^|\n)\s*glucose\s*(?:\(random\))?\s*(\d{2,3})',
                r'glucose.*?value.*?(\d{2,3})',
                r'\n\s*glucose\s*\(random\)\s*(\d{2,3})',  # Matches tabular format
                r'(?:^|\n)\s*glucose.*?(\d{2,3})\s*(?:mg/dl|mg/dL)?',  # More flexible glucose match
                r'(?:^|\n|\s)glucose[^a-zA-Z\d]*(\d{2,3})',  # Start of line/after newline
                r'glucose\s*(?:level)?[^a-zA-Z\d]*(\d{2,3})',  # Handles "glucose level: 105"
                r'glucose\s*[-=:.]?\s*(\d{2,3})',  # Most flexible pattern
                r'blood\s+sugar\s*[-=:.]?\s*(\d{2,3})',  # Blood sugar variation
                r'(?:^|\n|\t)\s*glucose\s*[-=:.]?\s*(\d{2,3})',  # Strict tabular format
                r'glucose.*?(\d{2,3})\s*(?:mg/dl|mg/dL|mmol/L)?'  # Super flexible with units
            ],
            'urea': [
                r'(?:urea|bun)\s*(?:[-:=]|\s+)\s*(\d{2,3})',
                r'blood\s+urea\s*(?:[-:=]|\s+)\s*(\d{2,3})',
                r'(?:^|\n)\s*urea\s*(\d{2,3})',  # Matches tabular format
                r'\n\s*urea\s*(\d{1,3})',  # Matches "Urea 77" in table
                r'(?:^|\n|\s)urea[^a-zA-Z\d]*(\d{1,3})',  # Start of line/after newline
                r'urea\s*(?:nitrogen)?[^a-zA-Z\d]*(\d{1,3})',  # Handles "urea nitrogen: 77"
                r'urea\s*[-=:.]?\s*(\d{1,3})',  # Most flexible pattern
                r'blood\s+urea\s*[-=:.]?\s*(\d{1,3})',  # Blood urea variation
                r'(?:^|\n|\t)\s*urea\s*[-=:.]?\s*(\d{1,3})',  # Strict tabular format
                r'urea.*?(\d{1,3})\s*(?:mg/dl|mg/dL|mmol/L)?'  # Super flexible with units
            ],
            'creatinine': [
                r'creatinine\s*(?:[-:=]|\s+)\s*(\d+\.?\d*)',
                r'serum\s+creatinine\s*(?:[-:=]|\s+)\s*(\d+\.?\d*)',
                r'(?:^|\n)\s*creatinine\s*(\d+\.?\d*)',  # Matches tabular format
                r'\n\s*creatinine\s*(\d+\.?\d*)',  # Matches "Creatinine 2.2" in table
                r'(?:^|\n|\s)creatinine[^a-zA-Z\d]*(\d+\.?\d*)',  # Start of line/after newline
                r'creatinine\s*(?:level)?[^a-zA-Z\d]*(\d+\.?\d*)',  # Handles "creatinine level: 2.2"
                r'creatinine\s*[-=:.]?\s*(\d+\.?\d*)',  # Most flexible pattern
                r'serum\s+creatinine\s*[-=:.]?\s*(\d+\.?\d*)',  # Serum creatinine variation
                r'(?:^|\n|\t)\s*creatinine\s*[-=:.]?\s*(\d+\.?\d*)',  # Strict tabular format
                r'creatinine.*?(\d+\.?\d*)\s*(?:mg/dl|mg/dL|mmol/L)?'  # Super flexible with units
            ]
        }

        # Process basic patterns
        for field, patterns in basic_patterns.items():
            print(f"\nLooking for {field}:")
            for pattern in patterns:
                print(f"Trying pattern: {pattern}")
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    value = match.group(1).lower()
                    print(f"Found match: '{match.group(0)}' -> value: '{value}'")
                    if field == 'sex':
                        # Convert sex/gender to numeric (1 for male, 0 for female)
                        data['sex'] = 1 if value.startswith('m') else 0
                        print(f"Found {field}: {value} -> {data['sex']}")
                    elif field == 'age':
                        # Convert age to float and validate
                        try:
                            age = float(value)
                            if 20 <= age <= 100:  # Basic age validation
                                data['age'] = age
                                print(f"Found {field}: {data['age']}")
                                break
                        except ValueError:
                            continue
                    break

        # Process test result patterns
        print("\nExtracting test results:")
        for test, patterns in test_patterns.items():
            print(f"\nLooking for {test}:")
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        if test == 'glucose':
                            if 50 <= value <= 500:  # Reasonable glucose range
                                data['fbs'] = 1 if value > 120 else 0
                                print(f"Found glucose: {value} -> fbs: {data['fbs']}")
                        elif test == 'urea':
                            if 10 <= value <= 200:  # Reasonable urea range
                                data['urea'] = value  # Store urea value
                                print(f"Found urea: {value}")
                        elif test == 'creatinine':
                            if 0.1 <= value <= 15.0:  # Reasonable creatinine range
                                data['creatinine'] = value  # Store creatinine value
                                print(f"Found creatinine: {value}")
                        break
                    except (ValueError, IndexError):
                        continue

        print(f"\nExtracted data: {data}")
        return data if data else None

    def detect_report_type(self, text):
        """
        Detect whether the text is from an ECG report or diagnostic report.
        Returns 'ecg' or 'diagnostic'.
        """
        text_lower = text.lower()
        
        # ECG report indicators
        ecg_indicators = [
            r'(?:^|\s)(?:ecg|ekg|electrocardiogram)(?:\s|$)',
            r'(?:^|\s)(?:nsr|normal sinus rhythm)(?:\s|$)',
            r'(?:^|\s)(?:004|193|239).*?(?:rhythm|infarction|changes)',
            r'(?:^|\s)(?:lvh|left ventricular hypertrophy)(?:\s|$)',
            r'(?:^|\s)(?:st\s+segment|t\s+wave)(?:\s|$)'
        ]
        
        # Diagnostic report indicators
        diagnostic_indicators = [
            r'laboratory\s+report',
            r'lab(?:oratory)?\s+(?:no|number)',
            r'specimen\s+received',
            r'medical\s+record\s+no',
            r'referring\s+physician',
            r'test[s]?\s+value\s+unit',
            r'normal\s+values?',
            r'renal\s+function\s+tests?',
            r'complete\s+blood\s+count',
            r'serum\s+glucose',
            r'sarwar\s+foundation'
        ]
        
        # Count matches for each type
        ecg_matches = 0
        diagnostic_matches = 0
        
        # Check ECG indicators
        for pattern in ecg_indicators:
            if re.search(pattern, text_lower):
                print(f"Found ECG indicator: {pattern}")
                ecg_matches += 1
        
        # Check diagnostic indicators
        for pattern in diagnostic_indicators:
            if re.search(pattern, text_lower):
                print(f"Found diagnostic indicator: {pattern}")
                diagnostic_matches += 1
        
        # Determine type based on matches
        print(f"DEBUG: Detected {ecg_matches} ECG indicators and {diagnostic_matches} diagnostic indicators")
        
        if ecg_matches >= 2:  # Require at least 2 ECG indicators
            print("DEBUG: Detected ECG report format")
            return 'ecg'
        elif diagnostic_matches >= 3:  # Require at least 3 diagnostic indicators
            print(f"DEBUG: Detected diagnostic report format (matches: {diagnostic_matches})")
            return 'diagnostic'
        else:
            print("DEBUG: Could not determine report type")
            return 'diagnostic'  # Default to diagnostic if unsure

    def extract_fields(self, text):
        """
        Ultra-robust extraction of medical fields from OCR text.
        Extracts: Age, Sex, Cp, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal.
        Handles full names, abbreviations, and value normalization/mapping.
        """
        print("\n" + "="*80)
        print("DEBUG: OCR Extracted Text:")
        print("="*80)
        print(text)
        print("="*80)
        
        # Show each line separately for better debugging
        # Show each line separately for better debugging
        lines = text.split('\n')
        print("\nDEBUG: Text by lines:")
        for i, line in enumerate(lines):
            if line.strip():  # Only show non-empty lines
                print(f"Line {i+1}: '{line.strip()}'")
        
        # Show all numbers found in the text
        all_numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
        print(f"\nDEBUG: All numbers found in text: {all_numbers}")
        
        # Show all decimal numbers found in the text
        all_decimals = re.findall(r'\b(\d+\.\d+)\b', text)
        print(f"DEBUG: All decimal numbers found: {all_decimals}")
        
        # Show all lines containing "ST" or "Depression"
        st_lines = [line for line in lines if 'st' in line.lower() or 'depression' in line.lower()]
        if st_lines:
            print(f"DEBUG: Lines containing 'ST' or 'Depression': {st_lines}")
        
        print("="*80)
        
        data = {}
        lines = text.split('\n')
        
        # Field definitions with multiple possible names and patterns
        field_defs = {
            'Age': {
                'names': ['Age', 'AGE', 'Patient Age', 'Age (years)', 'Age in years'],
                'patterns': [
                    r'(\d{1,3})\s*(?:years?|yrs?)',
                    r'Age[^0-9]*(\d{1,3})',
                    r'(\d{1,3})\s*$'
                ]
            },
            'Sex': {
                'names': ['Sex', 'Gender', 'SEX', 'GENDER', 'Patient Sex', 'Patient Gender'],
                'patterns': [
                    r'(Male|Female|M|F|male|female|m|f)',  # Case-insensitive matching
                    r'(Male|Female|M|F)',
                    r'(male|female|m|f)'
                ]
            },
            'Cp': {
                'names': ['Chest Pain Type', 'Chest Pain', 'CP', 'cp', 'Chest Pain Type:', 'Chest Pain:'],
                'patterns': [
                    r'(Typical Angina|Atypical Angina|Non-anginal Pain|Asymptomatic|typical angina|atypical angina|non-anginal pain|asymptomatic)',  # Case-insensitive matching
                    r'(Typical Angina|Atypical Angina|Non-anginal Pain|Asymptomatic)',
                    r'(typical angina|atypical angina|non-anginal pain|asymptomatic)',
                    r'(Typical|Atypical|Non-anginal|Asymptomatic)'
                ]
            },
            'Trestbps': {
                'names': ['Resting Blood Pressure', 'Blood Pressure', 'BP', 'Trestbps', 'Resting BP'],
                'patterns': [
                    r'(\d{2,3})/\d{2,3}',  # 120/80 format
                    r'(\d{2,3})\s*mmHg',
                    r'(\d{2,3})\s*mm Hg',
                    r'(\d{2,3})\s*mm Hg}',  # Handle OCR typo with }
                    r'(\d{2,3})\s*mm Hg\)',  # Handle OCR typo with )
                    r'(\d{2,3})\s*$',  # Just the number
                    r'(\d{1,3})\s*$'  # Handle missing first digit
                ]
            },
            'Chol': {
                'names': ['Serum Cholesterol', 'Cholesterol', 'Chol', 'CHOL', 'Total Cholesterol'],
                'patterns': [
                    r'(\d{2,4})\s*mg/dl',
                    r'(\d{2,4})\s*mg/dL',
                    r'(\d{2,4})\s*mg',
                    r'(\d{2,4})\s*$'  # Just the number
                ]
            },
            'Fbs': {
                'names': ['Fasting Blood Sugar', 'FBS', 'Fasting Glucose', 'Fasting Blood Sugar:', 'Fasting Blood Sugar'],
                'patterns': [
                    r'(\d{2,3})\s*mg/dl',
                    r'(\d{2,3})\s*mg/dL',
                    r'(\d{2,3})\s*mg',
                    r'(True|False)\s*\([^)]*\)',  # True/False with description
                    r'(True|False)',  # Just True/False
                    r'(>|≤)\s*120',  # > 120 or ≤ 120
                    r'(High|Normal)',  # High/Normal
                    r'(≤|<=)\s*120\s*mg/dl\s*\(Normal\)',  # ≤ 120 mg/dl (Normal)
                    r'(>|>)\s*120\s*mg/dl\s*\(High\)',  # > 120 mg/dl (High)
                    r'(\d{2,3})\s*mg/dl\s*\(Normal\)',  # 120 mg/dl (Normal)
                    r'(\d{2,3})\s*mg/dl\s*\(High\)'  # 150 mg/dl (High)
                ]
            },
            'Restecg': {
                'names': ['Resting ECG', 'RESTECG', 'ECG', 'Resting ECG:'],
                'patterns': [
                    r'(Normal|ST-T Wave Abnormality|LVH|normal|st-t wave abnormality|lvh)',  # Case-insensitive matching
                    r'(Normal|ST-T Wave Abnormality|LVH)',
                    r'(normal|st-t wave abnormality|lvh)',
                    r'(ST-T|ST-T Wave|ST-T Changes)',
                    r'(ST-T wave abnormality)',  # Exact match
                    r'(st-t wave abnormality)'  # Lowercase exact match
                ]
            },
            'Thalach': {
                'names': ['Maximum Heart Rate', 'Max Heart Rate', 'Heart Rate', 'Thalach', 'Max HR', 'Maximum Heart Rate:'],
                'patterns': [
                    r'(\d{2,3})\s*bpm',
                    r'(\d{2,3})\s*beats',
                    r'(\d{2,3})\s*beats per minute',
                    r'(\d{2,3})\s*$',
                    r'(\d{2,3})'  # Just the number if no units
                ]
            },
            'Exang': {
                'names': ['Exercise Induced Angina', 'Exang', 'Exercise Angina', 'Exercise Induced Angina:', 'Exercise Induced Angina'],
                'patterns': [
                    r'(Yes|No|Y|N|yes|no|y|n)',  # Case-insensitive matching
                    r'(Yes|No|Y|N)',
                    r'(yes|no|y|n)',
                    r'(1|0)',
                    r'(True|False|true|false)'  # Case-insensitive matching
                ]
            },
            'Oldpeak': {
                'names': ['ST Depression', 'ST Segment Depression', 'Oldpeak', 'ST Depression:', 'ST Depression', 'ST Depression', 'OT Depression', 'OT Segment Depression'],
                'patterns': [
                    r'(\d+\.?\d*)\s*mm',
                    r'(\d+\.?\d*)\s*mm Hg',
                    r'(\d+\.?\d*)\s*$',
                    r'(\d+\.?\d*)'  # Just the number if no units
                ]
            },
            'Slope': {
                'names': ['ST Segment Slope', 'ST Slope', 'Slope', 'ST Segment Slope:'],
                'patterns': [
                    r'(Upsloping|Flat|Downsloping|upsloping|flat|downsloping)',  # Case-insensitive matching
                    r'(Upsloping|Flat|Downsloping)',
                    r'(upsloping|flat|downsloping)',
                    r'(Up|Flat|Down)'
                ]
            },
            'Ca': {
                'names': ['Major Vessels', 'Number of Major Vessels', 'CA', 'Major Vessels:'],
                'patterns': [
                    r'(\d+)',
                    r'(\d+)\s*vessels'
                ]
            },
            'Thal': {
                'names': ['Thalassemia', 'Thal', 'THAL', 'Thalassemia:'],
                'patterns': [
                    r'(Normal|Fixed Defect|Reversible Defect|normal|fixed defect|reversible defect)',  # Case-insensitive matching
                    r'(Normal|Fixed Defect|Reversible Defect)',
                    r'(normal|fixed defect|reversible defect)',
                    r'(Fixed|Reversible)'
                ]
            }
        }
        
        print("\n" + "="*80)
        print("DEBUG: FIELD EXTRACTION PROCESS")
        print("="*80)
        
        # For each field, try all possible extraction methods
        for field, field_info in field_defs.items():
            print(f"\n--- Processing Field: {field} ---")
            value = None
            extraction_method = "None"
            
            # Special debugging for Oldpeak field
            if field == 'Oldpeak':
                print(f"  🔍 DEBUG: Special debugging for Oldpeak field")
                print(f"  🔍 DEBUG: Looking for ST Depression in text")
                st_matches = re.findall(r'st.*depression', text, re.IGNORECASE)
                if st_matches:
                    print(f"  🔍 DEBUG: Found ST Depression mentions: {st_matches}")
                else:
                    print(f"  🔍 DEBUG: No ST Depression mentions found")
                
                # Look for OCR errors (OT instead of ST)
                ot_matches = re.findall(r'ot.*depression', text, re.IGNORECASE)
                if ot_matches:
                    print(f"  🔍 DEBUG: Found OT Depression mentions (OCR error): {ot_matches}")
                
                # Look for any decimal numbers
                decimals = re.findall(r'\b(\d+\.\d+)\b', text)
                print(f"  🔍 DEBUG: All decimal numbers in text: {decimals}")
                
                # Look for numbers followed by 'mm'
                mm_numbers = re.findall(r'(\d+\.?\d*)\s*mm', text, re.IGNORECASE)
                print(f"  🔍 DEBUG: Numbers followed by 'mm': {mm_numbers}")
                
                # Look for depression-related lines
                depression_lines = [line for line in lines if 'depression' in line.lower()]
                if depression_lines:
                    print(f"  🔍 DEBUG: Lines containing 'depression': {depression_lines}")
            
            # Method 1: Try regex patterns for each name
            for name in field_info['names']:
                for pattern in field_info['patterns']:
                    # Try multiple variations of the pattern
                    variations = [
                        rf'{name}[^0-9A-Za-z]*[:\-]?\s*{pattern}',
                        rf'{name}\s*[:\-]?\s*{pattern}',
                        rf'{name}[^0-9A-Za-z]*{pattern}',
                        rf'{name}\s*{pattern}'
                    ]
                    
                    for full_pattern in variations:
                        match = re.search(full_pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            extraction_method = f"REGEX: {name} -> {pattern}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break
                    if value:
                        break
                if value:
                    break
            
            # Method 2: Line-by-line scanning if regex failed
            if not value:
                for line in lines:
                    line_lower = line.lower()
                    for name in field_info['names']:
                        if name.lower() in line_lower:
                            print(f"  🔍 Found '{name}' in line: '{line}'")
                            # Try to extract value after colon, dash, or space
                            parts = re.split(r'[:\-]', line, maxsplit=1)
                            if len(parts) > 1:
                                val_candidate = parts[1].strip()
                            else:
                                val_candidate = line.strip()
                            
                            # Try each pattern on the candidate value
                            for pattern in field_info['patterns']:
                                val_match = re.search(pattern, val_candidate, re.IGNORECASE)
                                if val_match:
                                    value = val_match.group(1).strip()
                                    extraction_method = f"LINE: {name} in '{line[:50]}...'"
                                    print(f"  ✓ Found via {extraction_method}: {value}")
                                    break
                            if value:
                                break
                    if value:
                        break
            
            # Method 3: Direct pattern matching in text if still no value
            if not value:
                for pattern in field_info['patterns']:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        extraction_method = f"DIRECT: {pattern}"
                        print(f"  ✓ Found via {extraction_method}: {value}")
                        break
            
            # Method 4: Context-aware extraction for specific fields
            if not value and field in ['Thalach', 'Oldpeak']:
                print(f"  🔍 Attempting context-aware extraction for {field}")
                if field == 'Thalach':
                    # Look for heart rate patterns in the text
                    heart_rate_patterns = [
                        r'(\d{2,3})\s*bpm',
                        r'(\d{2,3})\s*beats',
                        r'(\d{2,3})\s*beats per minute',
                        r'(\d{2,3})\s*$'
                    ]
                    for pattern in heart_rate_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            extraction_method = f"CONTEXT: Heart rate pattern -> {pattern}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break
                
                elif field == 'Oldpeak':
                    print(f"  🔍 DEBUG: Looking for ST Depression patterns in text")
                    # Look for ST depression patterns in the text
                    st_depression_patterns = [
                        r'(\d+\.?\d*)\s*mm',
                        r'(\d+\.?\d*)\s*mm Hg',
                        r'(\d+\.?\d*)\s*$'
                    ]
                    for pattern in st_depression_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            extraction_method = f"CONTEXT: ST depression pattern -> {pattern}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break

            
            # Method 5: Specific pattern matching for known report formats
            if not value and field in ['Thalach', 'Oldpeak', 'Exang']:
                print(f"  🔍 Attempting specific pattern matching for {field}")
                if field == 'Thalach':
                    # Look for "Maximum Heart Rate: 165 bpm" format
                    specific_patterns = [
                        r'Maximum Heart Rate[:\s]*(\d{2,3})\s*bpm',
                        r'Max Heart Rate[:\s]*(\d{2,3})\s*bpm',
                        r'Heart Rate[:\s]*(\d{2,3})\s*bpm',
                        r'(\d{2,3})\s*bpm.*Heart Rate',
                        r'(\d{2,3})\s*bpm.*Maximum'
                    ]
                    for pattern in specific_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            extraction_method = f"SPECIFIC: {pattern}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break
                
                elif field == 'Oldpeak':
                    # Look for "ST Depression: 2.3 mm" format (including OCR errors)
                    specific_patterns = [
                        r'ST Depression[:\s]*(\d+\.?\d*)\s*mm',
                        r'ST Segment Depression[:\s]*(\d+\.?\d*)\s*mm',
                        r'OT Depression[:\s]*(\d+\.?\d*)\s*mm',  # Handle OCR error: ST -> OT
                        r'OT Segment Depression[:\s]*(\d+\.?\d*)\s*mm',  # Handle OCR error: ST -> OT
                        r'(\d+\.?\d*)\s*mm.*ST Depression',
                        r'(\d+\.?\d*)\s*mm.*ST Segment',
                        r'(\d+\.?\d*)\s*mm.*OT Depression',  # Handle OCR error: ST -> OT
                        r'(\d+\.?\d*)\s*mm.*OT Segment'  # Handle OCR error: ST -> OT
                    ]
                    for pattern in specific_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            extraction_method = f"SPECIFIC: {pattern}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break
                
                elif field == 'Exang':
                    # Look for "Exercise Induced Angina: Yes" format
                    specific_patterns = [
                        r'Exercise Induced Angina[:\s]*(Yes|No)',
                        r'Exercise Angina[:\s]*(Yes|No)',
                        r'(Yes|No)[:\s]*Exercise Induced Angina',
                        r'(Yes|No)[:\s]*Exercise Angina'
                    ]
                    for pattern in specific_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            extraction_method = f"SPECIFIC: {pattern}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break
            
            # Method 6: Final fallback - look for any reasonable values
            if not value and field in ['Thalach', 'Oldpeak']:
                print(f"  🔍 Attempting final fallback extraction for {field}")
                if field == 'Thalach':
                    # Look for any 3-digit number that could be heart rate (100-250 range)
                    all_numbers = re.findall(r'\b(\d{3})\b', text)
                    for num in all_numbers:
                        num_val = int(num)
                        if 100 <= num_val <= 250:  # Reasonable heart rate range
                            value = str(num_val)
                            extraction_method = f"FALLBACK: Heart rate range validation -> {num_val}"
                            print(f"  ✓ Found via {extraction_method}: {value}")
                            break
                
                elif field == 'Oldpeak':
                    print(f"  🔍 DEBUG: Final fallback for Oldpeak - searching all decimal numbers")
                    # Look for any decimal number that could be ST depression (0.0-10.0 range)
                    all_decimals = re.findall(r'\b(\d+\.\d+)\b', text)
                    print(f"  🔍 DEBUG: Found decimal numbers: {all_decimals}")
                    
                    # First, try to find a decimal that's specifically associated with depression
                    depression_decimals = []
                    for line in lines:
                        if 'depression' in line.lower():
                            decimals_in_line = re.findall(r'\b(\d+\.\d+)\b', line)
                            depression_decimals.extend(decimals_in_line)
                    
                    if depression_decimals:
                        print(f"  🔍 DEBUG: Decimals found in depression lines: {depression_decimals}")
                        for dec in depression_decimals:
                            dec_val = float(dec)
                            if 0.0 <= dec_val <= 10.0:  # Reasonable ST depression range
                                value = str(dec_val)
                                extraction_method = f"FALLBACK: Depression-associated decimal -> {dec_val}"
                                print(f"  ✓ Found via {extraction_method}: {value}")
                                break
                    
                    # If no depression-associated decimal found, try any decimal in range
                    if not value:
                        for dec in all_decimals:
                            dec_val = float(dec)
                            print(f"  🔍 DEBUG: Checking decimal {dec_val}")
                            if 0.0 <= dec_val <= 10.0:  # Reasonable ST depression range
                                value = str(dec_val)
                                extraction_method = f"FALLBACK: ST depression range validation -> {dec_val}"
                                print(f"  ✓ Found via {extraction_method}: {value}")
                                break
                    
                    # If still no value, try to find any decimal number that's not already used
                    if not value:
                        print(f"  🔍 DEBUG: No suitable decimal found, trying any decimal number")
                        # Get all decimal numbers and find one that's not used by other fields
                        used_decimals = []
                        
                        # Check what decimals are already used by other fields
                        for other_field, other_value in data.items():
                            if other_value is not None and isinstance(other_value, (int, float)):
                                used_decimals.append(float(other_value))
                        
                        print(f"  🔍 DEBUG: Used decimals: {used_decimals}")
                        print(f"  🔍 DEBUG: Available decimals: {all_decimals}")
                        
                        for dec in all_decimals:
                            dec_val = float(dec)
                            if dec_val not in used_decimals:
                                value = str(dec_val)
                                extraction_method = f"FALLBACK: Unused decimal -> {dec_val}"
                                print(f"  ✓ Found via {extraction_method}: {value}")
                                break
            
            # Normalize and map the value
            if value is not None:
                val_lc = value.lower() if isinstance(value, str) else str(value).lower()
                print(f"  📝 Raw value: '{value}' -> Normalized: '{val_lc}'")
                
                # Map categorical values
                if field == 'Sex':
                    if val_lc in ['m', 'male']:
                        value = 1
                    elif val_lc in ['f', 'female']:
                        value = 0
                elif field == 'Cp':
                    if 'typical' in val_lc and 'atypical' not in val_lc:
                        value = 0
                    elif 'atypical' in val_lc:
                        value = 1
                    elif 'non-anginal' in val_lc:
                        value = 2
                    elif 'asymptomatic' in val_lc:
                        value = 3
                    else:
                        value = None
                elif field == 'Fbs':
                    # First check if the original text contains dropdown format
                    original_text_lower = text.lower()
                    fbs_line = None
                    for line in lines:
                        if 'fasting blood sugar' in line.lower():
                            fbs_line = line
                            break
                    
                    if fbs_line:
                        fbs_line_lower = fbs_line.lower()
                        print(f"  🔢 FBS line found: '{fbs_line}'")
                        
                        # Check for dropdown format first
                        if '> 120' in fbs_line_lower or '>120' in fbs_line_lower:
                            value = 1
                            print(f"  🔢 FBS dropdown indicates HIGH (> 120) -> mapped to: {value}")
                        elif '≤ 120' in fbs_line_lower or '<= 120' in fbs_line_lower:
                            value = 0
                            print(f"  🔢 FBS dropdown indicates NORMAL (≤ 120) -> mapped to: {value}")
                        else:
                            # Try to extract numeric value
                            try:
                                fbs_num = float(re.findall(r'\d+', fbs_line)[0])
                                print(f"  🔢 FBS numeric value: {fbs_num}")
                                
                                # Handle different numeric values based on dropdown options
                                if fbs_num <= 120:
                                    value = 0  # Normal (≤ 120 mg/dl)
                                    print(f"  🔢 FBS {fbs_num} <= 120 -> mapped to: {value} (Normal)")
                                elif fbs_num > 120:
                                    value = 1  # High (> 120 mg/dl)
                                    print(f"  🔢 FBS {fbs_num} > 120 -> mapped to: {value} (High)")
                                else:
                                    value = None
                            except Exception:
                                # Handle text values like "True (> 120 mg/dl)" or dropdown text
                                val_lc = value.lower()
                                print(f"  🔢 FBS text value: '{value}' -> lowercase: '{val_lc}'")
                                
                                # Check for various text representations
                                if any(x in val_lc for x in ['true', '> 120', 'high', '>120', '> 120 mg/dl (high)']):
                                    value = 1
                                    print(f"  🔢 FBS text indicates HIGH -> mapped to: {value}")
                                elif any(x in val_lc for x in ['false', '<= 120', 'normal', '≤ 120', '≤120', '≤ 120 mg/dl (normal)']):
                                    value = 0
                                    print(f"  🔢 FBS text indicates NORMAL -> mapped to: {value}")
                                else:
                                    value = None
                                    print(f"  🔢 FBS text value not recognized: '{value}'")
                    else:
                        # Fallback to original logic
                        try:
                            # First try to extract numeric value
                            fbs_num = float(re.findall(r'\d+', value)[0])
                            print(f"  🔢 FBS numeric value: {fbs_num}")
                            
                            # Handle different numeric values based on dropdown options
                            if fbs_num <= 120:
                                value = 0  # Normal (≤ 120 mg/dl)
                                print(f"  🔢 FBS {fbs_num} <= 120 -> mapped to: {value} (Normal)")
                            elif fbs_num > 120:
                                value = 1  # High (> 120 mg/dl)
                                print(f"  🔢 FBS {fbs_num} > 120 -> mapped to: {value} (High)")
                            else:
                                value = None
                        except Exception:
                            # Handle text values like "True (> 120 mg/dl)" or dropdown text
                            val_lc = value.lower()
                            print(f"  🔢 FBS text value: '{value}' -> lowercase: '{val_lc}'")
                            
                            # Check for various text representations
                            if any(x in val_lc for x in ['true', '> 120', 'high', '>120', '> 120 mg/dl (high)']):
                                value = 1
                                print(f"  🔢 FBS text indicates HIGH -> mapped to: {value}")
                            elif any(x in val_lc for x in ['false', '<= 120', 'normal', '≤ 120', '≤120', '≤ 120 mg/dl (normal)']):
                                value = 0
                                print(f"  🔢 FBS text indicates NORMAL -> mapped to: {value}")
                            else:
                                value = None
                                print(f"  🔢 FBS text value not recognized: '{value}'")
                elif field == 'Restecg':
                    val_lc = value.lower()
                    if 'normal' in val_lc:
                        value = 0
                    elif any(x in val_lc for x in ['st-t', 'abnormal', 'changes', 'wave abnormality', 'st-t wave']):
                        value = 1
                    elif any(x in val_lc for x in ['lvh', 'left ventricular hypertrophy']):
                        value = 2
                    else:
                        value = None
                elif field == 'Exang':
                    if val_lc in ['yes', 'true', '1', 'y']:
                        value = 1
                    elif val_lc in ['no', 'false', '0', 'n']:
                        value = 0
                    else:
                        value = None
                elif field == 'Slope':
                    if 'up' in val_lc:
                        value = 0
                    elif 'flat' in val_lc:
                        value = 1
                    elif 'down' in val_lc:
                        value = 2
                    else:
                        value = None
                elif field == 'Thal':
                    if 'normal' in val_lc:
                        value = 1
                    elif 'fixed' in val_lc:
                        value = 2
                    elif 'reversible' in val_lc:
                        value = 3
                    else:
                        value = None
                elif field in ['Age', 'Trestbps', 'Chol', 'Thalach', 'Oldpeak', 'Ca']:
                    try:
                        value = float(re.findall(r'\d+\.?\d*', value)[0])
                        # Special validation for Trestbps (blood pressure)
                        if field == 'Trestbps' and value < 50:
                            # Likely missing first digit, correct to reasonable BP
                            corrected_value = value + 100  # 30 -> 130
                            print(f"  🔧 Correcting likely OCR error in BP: {value} -> {corrected_value}")
                            value = corrected_value
                        # Special validation for Thalach (heart rate)
                        elif field == 'Thalach':
                            if value < 50 or value > 250:
                                print(f"  ⚠️  Heart rate value {value} seems unreasonable, but keeping it")
                        # Special validation for Oldpeak (ST depression)
                        elif field == 'Oldpeak':
                            if value < 0 or value > 10:
                                print(f"  ⚠️  ST depression value {value} seems unreasonable, but keeping it")
                    except Exception:
                        value = None
                
                print(f"  ✅ Final mapped value: {value}")
            else:
                print(f"  ❌ No value found for {field}")
            
            data[field.lower()] = value
        
        # Add empty values for fields not found
        ui_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for key in ui_fields:
            if key not in data:
                data[key] = None
        
        print("\n" + "="*80)
        print("DEBUG: FINAL EXTRACTED DATA")
        print("="*80)
        
        # Show summary of what was found and what was missing
        found_fields = []
        missing_fields = []
        for field, value in data.items():
            if value is not None:
                found_fields.append(f"{field}: {value}")
            else:
                missing_fields.append(field)
        
        print(f"\n✅ FOUND ({len(found_fields)}): {', '.join(found_fields)}")
        if missing_fields:
            print(f"❌ MISSING ({len(missing_fields)}): {', '.join(missing_fields)}")
        else:
            print("🎉 ALL FIELDS EXTRACTED SUCCESSFULLY!")
        
                    # Special attention to Thalach and Oldpeak
            if 'thalach' in missing_fields:
                print("⚠️  WARNING: Thalach (Maximum Heart Rate) not found!")
            if 'oldpeak' in missing_fields:
                print("⚠️  WARNING: Oldpeak (ST Depression) not found!")
                print("🔧 ATTEMPTING FORCED EXTRACTION FOR OLDPEAK...")
                
                # Force extraction by looking for any decimal number that could be ST depression
                all_decimals = re.findall(r'\b(\d+\.\d+)\b', text)
                print(f"🔧 Found decimal numbers in text: {all_decimals}")
                
                # Find a decimal that's not already used by other fields
                used_values = []
                for field, val in data.items():
                    if val is not None:
                        try:
                            used_values.append(float(val))
                        except:
                            pass
                
                print(f"🔧 Used values: {used_values}")
                
                for dec in all_decimals:
                    dec_val = float(dec)
                    if dec_val not in used_values and 0.0 <= dec_val <= 10.0:
                        data['oldpeak'] = dec_val
                        print(f"🔧 FORCED EXTRACTION: Set oldpeak to {dec_val}")
                        break
                else:
                    # If no suitable decimal found, use the first available decimal
                    if all_decimals:
                        dec_val = float(all_decimals[0])
                        data['oldpeak'] = dec_val
                        print(f"🔧 FORCED EXTRACTION: Set oldpeak to first available decimal {dec_val}")
                    else:
                        print("🔧 FORCED EXTRACTION: No decimal numbers found in text!")
        
        print("="*80)
        print("="*80)
        for key, value in data.items():
            print(f"  {key}: {value}")
        print("="*80)
        
        return data
    
    def is_ecg_report(self, image_path):
        """Check if the image is an ECG report"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Look for grid pattern characteristic of ECG
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Count strong edges
            strong_edges = np.sum(magnitude > np.mean(magnitude) + np.std(magnitude))
            
            # If there are many strong edges in a grid pattern, likely an ECG
            return strong_edges > (img.shape[0] * img.shape[1] * 0.1)
        except Exception as e:
            print(f"Error checking if image is ECG report: {str(e)}")
            return False
    
    def process_ecg_image(self, image_path):
        """
        Process an ECG image to extract cardiac features.
        
        Parameters:
        - image_path: Path to the ECG image
        
        Returns:
        - Dictionary of extracted ECG features
        """
        try:
            # Initialize ECG analyzer if not already done
            if not hasattr(self, 'ecg_analyzer'):
                self.ecg_analyzer = ECGAnalyzer()

            # Load and preprocess the ECG image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = image_path

            if image is None:
                raise ValueError("Failed to load ECG image")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance grid lines and waveforms
            enhanced = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Extract grid parameters
            grid_size = self._detect_grid_size(enhanced)
            print(f"Detected grid size: {grid_size}")

            # Analyze the ECG using our analyzer
            results = self.ecg_analyzer.analyze_ecg(enhanced)
            
            # Create a visualization with the analysis
            if isinstance(image_path, str):
                visualization_path = os.path.splitext(image_path)[0] + "_analyzed.png"
                self.ecg_analyzer.visualize_analysis(image, save_path=visualization_path)

            # Extract the key features we need for heart disease prediction
            extracted_data = {}
            
            # Aggregate results from all leads
            avg_heart_rate = []
            avg_qt_interval = []
            avg_qtc_interval = []
            avg_st_deviation = []
            avg_pr_interval = []
            avg_qrs_duration = []
            
            for lead, lead_results in results.items():
                if lead_results.get('heart_rate'):
                    avg_heart_rate.append(lead_results['heart_rate'])
                if lead_results.get('qt_interval'):
                    avg_qt_interval.append(lead_results['qt_interval'])
                if lead_results.get('qtc_interval'):
                    avg_qtc_interval.append(lead_results['qtc_interval'])
                if lead_results.get('st_deviation'):
                    avg_st_deviation.append(lead_results['st_deviation'])
                if lead_results.get('pr_interval'):
                    avg_pr_interval.append(lead_results['pr_interval'])
                if lead_results.get('qrs_duration'):
                    avg_qrs_duration.append(lead_results['qrs_duration'])

            # Calculate average values across leads
            if avg_heart_rate:
                extracted_data['thalach'] = float(np.mean(avg_heart_rate))
            
            # Analyze ST segment for oldpeak and slope
            if avg_st_deviation:
                mean_st = float(np.mean(avg_st_deviation))
                if abs(mean_st) > 0.05:  # 0.5mm threshold for significant ST changes
                    extracted_data['oldpeak'] = abs(mean_st)
                    
                    # Determine slope of ST segment
                    slopes = self._analyze_st_slopes(results)
                    if slopes:
                        # 0 = upsloping, 1 = flat, 2 = downsloping
                        slope_counts = {0: 0, 1: 0, 2: 0}
                        for slope in slopes:
                            slope_counts[slope] += 1
                        extracted_data['slope'] = max(slope_counts, key=slope_counts.get)

            # Analyze ECG waves for other cardiac features
            if avg_qt_interval:
                mean_qtc = float(np.mean(avg_qtc_interval)) if avg_qtc_interval else 0
                # QTc > 450ms in men or > 470ms in women indicates abnormal repolarization
                extracted_data['restecg'] = 1 if mean_qtc > 460 else 0

            # Look for evidence of exercise-induced changes
            if self._detect_exercise_changes(results):
                extracted_data['exang'] = 1
            else:
                extracted_data['exang'] = 0

            # Add confidence scores
            confidence_scores = self._calculate_ecg_confidence(results)
            extracted_data['confidence_scores'] = confidence_scores

            return extracted_data

        except Exception as e:
            print(f"Error processing ECG image: {str(e)}")
            traceback.print_exc()
            return {}

    def _detect_grid_size(self, image):
        """
        Detect the ECG grid size in pixels.
        """
        try:
            # Use Hough transform to detect grid lines
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return None

            # Calculate distances between parallel lines
            distances = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    
                    # Check if lines are parallel
                    if abs((y2-y1)/(x2-x1) - (y4-y3)/(x4-x3)) < 0.1:
                        distance = abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                        if distance > 5:  # Minimum distance threshold
                            distances.append(distance)

            if distances:
                # Find most common distance (grid size)
                grid_size = np.median(distances)
                return int(round(grid_size))
            
            return None

        except Exception as e:
            print(f"Error detecting grid size: {str(e)}")
            return None

    def _analyze_st_slopes(self, results):
        """
        Analyze ST segment slopes from ECG results.
        Returns list of slope classifications (0=up, 1=flat, 2=down)
        """
        slopes = []
        for lead_results in results.values():
            if 'st_segment' in lead_results:
                st_segment = lead_results['st_segment']
                if len(st_segment) >= 2:
                    # Calculate slope
                    slope = (st_segment[-1] - st_segment[0]) / len(st_segment)
                    # Classify slope
                    if slope > 0.1:
                        slopes.append(0)  # Upsloping
                    elif slope < -0.1:
                        slopes.append(2)  # Downsloping
                    else:
                        slopes.append(1)  # Flat
        return slopes

    def _analyze_ecg_waveform(self, image):
        """
        Analyze ECG waveform to extract specific heart disease prediction parameters.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhance contrast
            enhanced = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Detect grid size and calibrate measurements
            grid_size = self._detect_grid_size(enhanced)
            if not grid_size:
                grid_size = 25  # Default 25 pixels per 1mm

            # Extract waveform
            waveform = self._extract_waveform(enhanced)
            if not waveform:
                return {}

            extracted_data = {}

            # 1. Calculate Heart Rate (thalach)
            r_peaks = self._detect_r_peaks(waveform)
            if r_peaks and len(r_peaks) > 1:
                # Calculate average R-R interval
                rr_intervals = np.diff(r_peaks)
                mean_rr = np.mean(rr_intervals)
                # Convert to heart rate (assuming 25mm/s paper speed)
                heart_rate = 60 * 25 / (mean_rr / grid_size)
                if 60 <= heart_rate <= 200:
                    extracted_data['thalach'] = float(heart_rate)

            # 2. Analyze ST Segment
            st_measurements = self._analyze_st_segment(waveform, r_peaks, grid_size)
            if st_measurements:
                # ST depression (oldpeak)
                if 0 <= st_measurements['depression'] <= 6.2:
                    extracted_data['oldpeak'] = float(st_measurements['depression'])
                # ST slope
                extracted_data['slope'] = st_measurements['slope']

            # 3. Analyze overall ECG pattern (restecg)
            ecg_class = self._classify_ecg_pattern(waveform, r_peaks, grid_size)
            if ecg_class is not None:
                extracted_data['restecg'] = ecg_class

            # 4. Check for exercise-induced changes
            if self._detect_exercise_changes(waveform, r_peaks, grid_size):
                extracted_data['exang'] = 1
            else:
                extracted_data['exang'] = 0

            return extracted_data
            
        except Exception as e:
            print(f"Error analyzing ECG waveform: {str(e)}")
            traceback.print_exc()
            return {}

    def _extract_waveform(self, image):
        """Extract ECG waveform from image."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            
            if not contours:
                return None 

            # Filter contours by size
            valid_contours = []
            for contour in contours:
                if len(contour) > 50:  # Minimum points for valid waveform
                    valid_contours.append(contour)

            if not valid_contours:
                return None

            # Sort points by x-coordinate
            all_points = np.concatenate(valid_contours)
            sorted_points = all_points[all_points[:, 0, 0].argsort()]
            
            return sorted_points

        except Exception as e:
            print(f"Error extracting waveform: {str(e)}")
            return None

    def _detect_r_peaks(self, waveform):
        """Detect R peaks in ECG waveform."""
        try:
            if waveform is None:
                return None

            # Convert to 1D signal
            signal = waveform[:, 0, 1]  # y-coordinates
            
            # Find peaks
            peaks, _ = signal.find_peaks(
                signal,
                distance=50,  # Minimum distance between peaks
                prominence=20  # Minimum prominence for R peaks
            )
            
            return peaks

        except Exception as e:
            print(f"Error detecting R peaks: {str(e)}")
            return None

    def _analyze_st_segment(self, waveform, r_peaks, grid_size):
        """
        Analyze ST segment characteristics.
        Returns depression depth and slope type.
        """
        try:
            if waveform is None or r_peaks is None:
                return None

            results = {
                'depression': 0.0,
                'slope': 1  # Default to flat
            }

            # For each R peak, analyze the following ST segment
            st_depressions = []
            st_slopes = []

            for peak in r_peaks:
                # Define ST segment region (80-120ms after R peak)
                st_start = peak + int(0.08 * grid_size * 25)  # 80ms
                st_end = peak + int(0.12 * grid_size * 25)    # 120ms
                
                if st_end >= len(waveform):
                    continue

                # Get ST segment points
                st_segment = waveform[st_start:st_end, 0, 1]
                
                # Calculate depression (convert pixels to mm)
                baseline = np.mean(waveform[peak-10:peak-5, 0, 1])
                depression = (baseline - np.mean(st_segment)) / grid_size
                st_depressions.append(depression)

                # Calculate slope
                if len(st_segment) > 2:
                    slope = (st_segment[-1] - st_segment[0]) / len(st_segment)
                    if slope > 0.1:
                        st_slopes.append(0)  # Upsloping
                    elif slope < -0.1:
                        st_slopes.append(2)  # Downsloping
                    else:
                        st_slopes.append(1)  # Flat

            if st_depressions:
                results['depression'] = float(np.mean(st_depressions))
            if st_slopes:
                # Use most common slope type
                from collections import Counter
                results['slope'] = Counter(st_slopes).most_common(1)[0][0]

            return results

        except Exception as e:
            print(f"Error analyzing ST segment: {str(e)}")
            return None

    def _classify_ecg_pattern(self, waveform, r_peaks, grid_size):
        """
        Classify ECG pattern into normal, ST-T abnormality, or LVH.
        Returns 0 (normal), 1 (ST-T abnormality), or 2 (LVH)
        """
        try:
            if waveform is None or r_peaks is None:
                return None

            # Check for ST-T abnormalities
            st_measurements = self._analyze_st_segment(waveform, r_peaks, grid_size)
            if st_measurements and abs(st_measurements['depression']) > 0.1:
                return 1  # ST-T abnormality

            # Check for LVH (high R wave amplitude)
            max_r_amplitude = 0
            for peak in r_peaks:
                amplitude = waveform[peak, 0, 1] / grid_size  # Convert to mm
                max_r_amplitude = max(max_r_amplitude, amplitude)

            if max_r_amplitude > 20:  # More than 20mm in any lead
                return 2  # Probable LVH

            return 0  # Normal

        except Exception as e:
            print(f"Error classifying ECG pattern: {str(e)}")
            return None

    def _detect_exercise_changes(self, results):
        """
        Detect changes suggestive of exercise-induced ischemia.
        Returns True if changes detected, False otherwise.
        """
        try:
            if results is None:
                return False

            # Check for significant ST depression
            st_measurements = self._analyze_st_segment(results)
            if st_measurements and st_measurements['depression'] > 0.1:
                return True

            # Check for T wave inversion
            for lead_results in results.values():
                if 't_wave' in lead_results:
                    t_wave = lead_results['t_wave']
                    if len(t_wave) > 0:
                        t_amplitude = (np.max(t_wave) - np.min(t_wave)) / grid_size
                        if t_amplitude < -0.1:  # Inverted T wave
                            return True

            return False

        except Exception as e:
            print(f"Error detecting exercise changes: {str(e)}")
            return False

    def _calculate_ecg_confidence(self, results):
        """
        Calculate confidence scores for ECG analysis results
        """
        confidence = {
            'heart_rate': 0.0,
            'st_analysis': 0.0,
            'wave_detection': 0.0
        }
        
        valid_leads = 0
        for lead_results in results.values():
            if lead_results.get('quality_score', 0) > 0.7:
                valid_leads += 1
                if lead_results.get('heart_rate'):
                    confidence['heart_rate'] += 1
                if lead_results.get('st_deviation') is not None:
                    confidence['st_analysis'] += 1
                if all(k in lead_results for k in ['p_wave', 'qrs_complex', 't_wave']):
                    confidence['wave_detection'] += 1
        
        if valid_leads > 0:
            confidence = {k: v/valid_leads for k, v in confidence.items()}
            
        return confidence 

    def process_report(self, file_path):
        """Process a file (PDF or image) and extract data"""
        try:
            print(f"\nProcessing file: {file_path}")
            extracted_data = None
            
            if self.is_pdf(file_path):
                # Handle PDF
                print("Processing PDF file")
                images = self.convert_pdf_to_images(file_path)
                for i, img in enumerate(images):
                    # Convert PIL Image to numpy array
                    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    # Process the image
                    processed_img = self.preprocess_image(img_array)
                    text = self.extract_text(processed_img)
                    
                    # Detect report type
                    if self.detect_report_type(text) == 'ecg':
                        page_data = self.extract_ecg_data(text)
                    else:
                        page_data = self.extract_fields(text)  # Use robust extraction
                    
                    if page_data:
                        # Validate the extracted data
                        extracted_data = self.validate_extracted_data(page_data)
                        if extracted_data:
                            extracted_data = extracted_data
                            break  # Use first page with valid data
            else:
                # Handle image file
                print("Processing image file")
                img = cv2.imread(file_path)
                if img is not None:
                    processed_img = self.preprocess_image(img)
                    text = self.extract_text(processed_img)
                    
                    # Detect report type
                    if self.detect_report_type(text) == 'ecg':
                        page_data = self.extract_ecg_data(text)
                    else:
                        page_data = self.extract_fields(text)  # Use robust extraction
                    
                    if page_data:
                        # Validate the extracted data
                        extracted_data = self.validate_extracted_data(page_data)
                else:
                    print(f"Could not read image: {file_path}")
            
            return extracted_data
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            traceback.print_exc()
            return None

    def _is_valid_value(self, field, value):
        """
        Validate if a value is within acceptable range for a given field
        """
        ranges = {
            'age': (20, 100),
            'thalach': (60, 200),
            'oldpeak': (0, 6.2),
            'restecg': (0, 2) # Added for restecg
        }
        
        if field in ranges:
            min_val, max_val = ranges[field]
            return min_val <= value <= max_val
        return True

    def extract_data(self, text):
        """Extract data from ECG report format."""
        data = {}
        text_lower = text.lower()
        
        print("\nDEBUG: Raw text for ECG extraction:")
        print(text)
        
        # ECG-specific patterns
        patterns = {
            'age': [
                r'date of birth:.*?\((\d{1,3})\s*years?\)',  # Match "(61 years)" after DOB
                r'age\s*[:.]\s*(\d{1,3})',  # Match "Age: 61"
                r'(\d{1,3})\s*(?:yrs?|years?|y\.?o\.?)'  # Match "61 years"
            ],
            'sex': [
                r'gender[:\s]*\n*\s*(male|female|m|f)\b',  # Match "Gender: \n Male"
                r'sex[:\s]*\n*\s*(male|female|m|f)\b',  # Match "Sex: \n Male"
                r'\n(male|female|m|f)\s*\[pos:',  # Match OCR format "Male [pos:"
            ],
            'thalach': [
                r'heart rate[:\s]*\n*\s*(\d{1,3})\s*(?:bpm|/min)?',  # Match "Heart Rate: \n 80 bpm"
                r'hr[:\s]*\n*\s*(\d{1,3})\s*(?:bpm|/min)?',  # Match "HR: \n 80 bpm"
                r'\n(\d{1,3})\s*bpm\s*\[pos:',  # Match OCR format "80 bpm [pos:"
            ],
            'restecg': [
                r'(?:^|\n)\s*004\s*(?:normal sinus rhythm|nsr)',  # Normal (0)
                r'(?:^|\n)\s*(?:st-t wave abnormality|st-t changes|st-t abnormal)',  # ST-T abnormality (1)
                r'(?:^|\n)\s*(?:lvh|left ventricular hypertrophy)',  # LVH (2)
                r'(?:^|\n)\s*193\s*(?:.*?infarction)',  # ST-T abnormality (1)
                r'(?:^|\n)\s*239\s*(?:.*?(?:st|st-t)\s*changes)',  # ST-T abnormality (1)
                r'(?:^|\n)\s*(?:anteroseptal infarction|ischemic st-t changes)',  # ST-T abnormality (1)
            ],
            'oldpeak': [
                r'st\s*segment\s*depression\s*[:.]\s*(\d+(?:\.\d+)?)',
                r'st\s*depression\s*[:.]\s*(\d+(?:\.\d+)?)',
                r'st[:\s]*\n*\s*(\d+(?:\.\d+)?)\s*(?:mm|µv/s)',  # Match "ST: \n 1.0 mm"
            ]
        }
        
        for field, field_patterns in patterns.items():
            print(f"\nDEBUG: Processing field: {field}")
            
            for pattern in field_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if field == 'sex':
                        value = match.group(1).strip().lower()
                        if value in ['male', 'm']:
                            data[field] = 1  # Integer, not float
                        elif value in ['female', 'f']:
                            data[field] = 0  # Integer, not float
                        print(f"DEBUG: Found match for {field}: {match.group(1)}")
                        print(f"DEBUG: Set {field} to {data[field]}")
                        break
                    
                    elif field == 'restecg':
                        matched_text = match.group(0).lower()
                        print(f"DEBUG: Found match for restecg: {matched_text}")
                        if '004' in matched_text and 'normal sinus rhythm' in matched_text:
                            data[field] = 0  # Normal
                            print("DEBUG: Set restecg to 0 (Normal)")
                        elif 'lvh' in matched_text or 'left ventricular hypertrophy' in matched_text:
                            data[field] = 2  # LVH
                            print("DEBUG: Set restecg to 2 (LVH)")
                        else:  # ST-T abnormality
                            data[field] = 1  # ST-T wave abnormality
                            print("DEBUG: Set restecg to 1 (ST-T Wave Abnormality)")
                        
                    else:  # age, thalach, oldpeak
                        try:
                            value = float(match.group(1))
                            if field == 'age' and not (0 < value < 120):
                                print(f"DEBUG: Invalid age value: {value}")
                                continue
                            data[field] = value
                            print(f"DEBUG: Found match for {field}: {match.group(1)}")
                            print(f"DEBUG: Set {field} to {value}")
                            break
                        except (ValueError, IndexError) as e:
                            print(f"DEBUG: Error converting {field} value: {e}")
                            continue
        
        # Ensure integer values for categorical fields
        if 'sex' in data:
            data['sex'] = int(data['sex'])
        if 'restecg' in data:
            data['restecg'] = int(data['restecg'])
            
        print("\nDEBUG: Final extracted ECG data:", data)
        return data 

    def extract_ecg_data(self, text):
        """Extract data from ECG report format."""
        data = {}
        text_lower = text.lower()
        
        print("\nDEBUG: Raw text for ECG extraction:")
        print(text)
        
        # Process fields in sequence
        fields = ['age', 'sex', 'thalach', 'restecg', 'oldpeak']
        
        for field in fields:
            print(f"\nDEBUG: Processing field: {field}")
            
            if field == 'age':
                # Look for age in date of birth
                dob_match = re.search(r'(?:date\s+of\s+birth|dob|birth\s+date)[^(]*\((\d+)\s*years?\)', text_lower)
                if dob_match:
                    age = float(dob_match.group(1))
                    print(f"DEBUG: Found match for age: {age}")
                    data['age'] = age
                    print(f"DEBUG: Set age to {data['age']}")
            
            elif field == 'sex':
                # Look for gender/sex
                sex_match = re.search(r'(?:gender|sex)\s*:\s*(male|female|m|f)\b', text_lower)
                if sex_match:
                    sex = sex_match.group(1).lower()
                    data['sex'] = 1 if sex.startswith('m') else 0
                    print(f"DEBUG: Found match for sex: {sex}")
                    print(f"DEBUG: Set sex to {data['sex']}")
            
            elif field == 'thalach':
                # Look for heart rate
                hr_match = re.search(r'heart\s+rate\s*:\s*(\d+)', text_lower)
                if hr_match:
                    thalach = float(hr_match.group(1))
                    print(f"DEBUG: Found match for thalach: {thalach}")
                    data['thalach'] = thalach
                    print(f"DEBUG: Set thalach to {data['thalach']}")
            
            elif field == 'restecg':
                # Look for ST-T wave abnormality indicators
                st_t_indicators = [
                    r'(?:^|\s)(?:st[-\s]t\s+wave\s+abnormal)',
                    r'(?:^|\s)(?:st[-\s]t\s+changes)',
                    r'(?:^|\s)(?:st\s+segment\s+abnormal)',
                    r'(?:^|\s)(?:t\s+wave\s+abnormal)',
                    r'(?:^|\s)(?:239.*?(?:st[-\s]t|ischemic)\s+changes)',
                ]
                
                # Look for each indicator
                for pattern in st_t_indicators:
                    if re.search(pattern, text_lower):
                        print(f"DEBUG: Found ST-T wave abnormality")
                        data['restecg'] = 1  # ST-T wave abnormality
                        break
                else:
                    print(f"DEBUG: No ST-T wave abnormality found")
                    data['restecg'] = 0  # Normal
            
            elif field == 'oldpeak':
                # Look for ST depression
                st_match = re.search(r'st\s+depression\s*[:-]?\s*(\d+\.?\d*)', text_lower)
                if st_match:
                    oldpeak = float(st_match.group(1))
                    print(f"DEBUG: Found match for oldpeak: {oldpeak}")
                    data['oldpeak'] = oldpeak
                    print(f"DEBUG: Set oldpeak to {data['oldpeak']}")
        
        print(f"\nDEBUG: Final extracted ECG data: {data}")
        return data if data else None 

