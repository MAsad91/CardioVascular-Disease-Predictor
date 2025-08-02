import os
import io
import base64
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from io import BytesIO

class ProfessionalPDFReport:
    """Professional PDF report generator for heart disease predictions"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.create_custom_styles()
    
    def create_custom_styles(self):
        """Create custom styles for the professional report"""
        # Main title style
        self.title_style = ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Title'],
            fontSize=16,
            textColor=colors.HexColor('#1f4e79'),
            spaceAfter=8,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        )
        
        # Report header info style
        self.header_info_style = ParagraphStyle(
            'HeaderInfo',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            spaceAfter=3,
            fontName='Helvetica'
        )
        
        # Risk level style (prominent)
        self.risk_level_style = ParagraphStyle(
            'RiskLevel',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#2e8b57'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        # Section heading style
        self.section_heading_style = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#1f4e79'),
            spaceAfter=6,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        # Subsection heading style
        self.subsection_heading_style = ParagraphStyle(
            'SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.HexColor('#1f4e79'),
            spaceAfter=4,
            spaceBefore=6,
            fontName='Helvetica-Bold'
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
        textColor=colors.black,
            spaceAfter=4,
            fontName='Helvetica'
        )
        
        # Recommendation bullet style
        self.bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=2,
            fontName='Helvetica'
        )
        
        # Disclaimer style
        self.disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
            parent=self.styles['Normal'],
        fontSize=8,
            textColor=colors.HexColor('#666666'),
            alignment=0,  # Left alignment
            spaceAfter=5,
            fontName='Helvetica-Oblique'
        )

    def create_page_1(self, explanation, input_data, feature_descriptions, session_id):
        """Create Page 1: Header, Risk Level, and Complete Patient Information"""
        story = []
        
        # Report Title
        story.append(Paragraph("Heart Disease Risk Assessment Report", self.title_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Report metadata
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        story.append(Paragraph(f"Report Date: {current_date}", self.header_info_style))
        story.append(Paragraph(f"Patient ID: {session_id[:8]}...", self.header_info_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Risk Level (prominent display with larger spacing)
        risk_level = explanation.get('risk_level', 'Unknown')
        probability = explanation.get('probability', 0.0)
        
        # Create colored box for risk level
        if risk_level.lower() == 'low':
            risk_color = colors.HexColor('#2e8b57')  # Green
        elif risk_level.lower() == 'medium':
            risk_color = colors.HexColor('#ff8c00')  # Orange
        else:
            risk_color = colors.HexColor('#dc143c')  # Red
            
        risk_style = ParagraphStyle(
            'RiskLevelColored',
            parent=self.risk_level_style,
            textColor=risk_color,
            fontSize=16,
            spaceAfter=10
        )
        
        story.append(Paragraph(f"Risk Level: {risk_level} ({probability:.0%})", risk_style))
        
        # Risk description with better spacing
        story.append(Paragraph(explanation.get('risk_description', 'Low risk of heart disease based on multiple AI models. Maintain a healthy lifestyle.'), self.body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Complete Patient Information Table (All fields included)
        story.append(Paragraph("Complete Health Parameters", self.section_heading_style))
        
        # Create comprehensive patient data table with ALL fields
        table_data = [["Health Parameter", "Value", "Assessment"]]
        
        # All field mappings with assessments
        all_field_mappings = {
            'age': ('Age', lambda x: f"{x:.0f} years", 
                   lambda x: "Normal" if 30 <= x <= 65 else "Higher Risk" if x > 65 else "Young"),
            'sex': ('Gender', lambda x: "Male" if x == 1 else "Female",
                   lambda x: "Higher Risk" if x == 1 else "Lower Risk"),
            'cp': ('Chest Pain Type', 
                  lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][int(x)],
                  lambda x: "High Risk" if x == 0 else "Moderate Risk" if x == 1 else "Low Risk" if x == 2 else "Asymptomatic"),
            'trestbps': ('Blood Pressure', lambda x: f"{x:.0f} mmHg",
                        lambda x: "Normal" if x < 120 else "Elevated" if x < 140 else "High"),
            'chol': ('Cholesterol', lambda x: f"{x:.0f} mg/dl",
                    lambda x: "Normal" if x < 200 else "Borderline" if x < 240 else "High"),
            'fbs': ('Fasting Blood Sugar', lambda x: "Yes" if x == 1 else "No",
                   lambda x: "High" if x == 1 else "Normal"),
            'restecg': ('Resting ECG', 
                       lambda x: ["Normal", "ST-T abnormality", "LV hypertrophy"][int(x)],
                       lambda x: "Normal" if x == 0 else "Abnormal"),
            'thalach': ('Max Heart Rate', lambda x: f"{x:.0f} bpm",
                       lambda x: "Good" if x > 150 else "Moderate" if x > 120 else "Low"),
            'exang': ('Exercise Angina', lambda x: "Yes" if x == 1 else "No",
                     lambda x: "High Risk" if x == 1 else "Normal"),
            'oldpeak': ('ST Depression', lambda x: f"{x:.1f}",
                       lambda x: "Normal" if x < 1 else "Mild" if x < 2 else "Significant"),
            'slope': ('ST Segment Slope', 
                     lambda x: ["Upsloping", "Flat", "Downsloping"][int(x)],
                     lambda x: "Good" if x == 0 else "Moderate" if x == 1 else "Poor"),
            'ca': ('Major Vessels', lambda x: f"{x:.0f}",
                  lambda x: "Normal" if x == 0 else "Moderate" if x <= 2 else "High Risk"),
            'thal': ('Thalassemia', 
                    lambda x: ["Normal", "Fixed defect", "Reversible defect"][int(x)-1] if x in [1,2,3] else "Unknown",
                    lambda x: "Normal" if x == 1 else "Abnormal" if x in [2,3] else "Unknown")
        }
        
        # Add data rows for ALL fields
        for field, (description, formatter, assessor) in all_field_mappings.items():
            if field in input_data:
                try:
                    formatted_value = formatter(input_data[field])
                    assessment = assessor(input_data[field])
                    table_data.append([description, formatted_value, assessment])
                except (IndexError, ValueError, TypeError):
                    table_data.append([description, str(input_data[field]), "Unknown"])
        
        # Create comprehensive table with proper styling
        table = Table(table_data, colWidths=[2.2*inch, 1.8*inch, 1.5*inch])
        table.setStyle(TableStyle([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#b8cce4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, 0), 4),
            
            # Data rows styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f2f7fb')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left align descriptions
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),  # Center align values
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),  # Center align assessments
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('TOPPADDING', (0, 1), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.15*inch))
        
        return story

    def create_page_2(self, explanation):
        """Create Page 2: Enhanced Visualizations and Analysis"""
        story = []
        
        # Your Key Risk Factors (Single display - main chart)
        story.append(Paragraph("Your Key Risk Factors", self.section_heading_style))
        
        if 'key_risk_factors_img' in explanation and explanation['key_risk_factors_img']:
            try:
                img_data = base64.b64decode(explanation['key_risk_factors_img'])
                img = Image(BytesIO(img_data), width=5.5*inch, height=3.5*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.08*inch))
            except Exception as e:
                story.append(Paragraph("Key risk factors visualization not available.", self.body_style))
        else:
            story.append(Paragraph("Key risk factors visualization not available.", self.body_style))
            
        story.append(Paragraph("This chart displays your risk factor scores (0-100 scale) with reference lines to help interpret the significance of each factor based on your health parameters.", self.body_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Feature Impact Analysis
        story.append(Paragraph("Detailed Feature Impact Analysis", self.section_heading_style))
        
        if 'risk_factor_img' in explanation and explanation['risk_factor_img']:
            try:
                img_data = base64.b64decode(explanation['risk_factor_img'])
                img = Image(BytesIO(img_data), width=5.5*inch, height=3.5*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.08*inch))
            except Exception as e:
                story.append(Paragraph("Feature impact analysis visualization not available.", self.body_style))
        else:
            story.append(Paragraph("Feature impact analysis visualization not available.", self.body_style))
            
        story.append(Paragraph("This enhanced chart shows how your health parameters contribute to your overall risk assessment. The scaled scores (20-100) represent the relative impact of each factor, with color coding for easy interpretation.", self.body_style))
        
        return story

    def create_page_3(self, explanation, input_data):
        """Create Page 3: Complete Patient Comparisons and Recommendations"""
        story = []
        
        # Complete Patient Comparison Section (ALL boxplots)
        story.append(Paragraph("Complete Comparison to Similar Patients", self.section_heading_style))
        
        # Add ALL available box plots
        if 'boxplots' in explanation and explanation['boxplots']:
            boxplot_count = 0
            
            # Show ALL boxplots available
            for feature, img_base64 in explanation['boxplots'].items():
                try:
                    img_data = base64.b64decode(img_base64)
                    # Use improved size to match web display
                    img = Image(BytesIO(img_data), width=8.0*inch, height=1.5*inch)
                    img.hAlign = 'CENTER'
                    story.append(img)
                    story.append(Spacer(1, 0.02*inch))
                    boxplot_count += 1
                except Exception as e:
                    continue
            
            if boxplot_count > 0:
                story.append(Paragraph(f"All {boxplot_count} available box plots above show how your values (red dots) compare to similar patients in our database for comprehensive analysis.", self.body_style))
                story.append(Spacer(1, 0.15*inch))
        else:
            story.append(Paragraph("Patient comparison data not available.", self.body_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Health Recommendations (compact layout)
        story.append(Paragraph("Personalized Health Recommendations", self.section_heading_style))
        
        risk_level = explanation.get('risk_level', 'Low').lower()
        
        if risk_level == 'high':
            recommendations = [
                "• <b>Immediate medical consultation</b> - Schedule appointment with healthcare provider",
                "• <b>Medication compliance</b> - Take prescribed medications as directed",
                "• <b>Emergency awareness</b> - Know heart attack signs and emergency procedures",
                "• <b>Lifestyle modifications</b> - Implement supervised diet and exercise plan"
            ]
        elif risk_level == 'medium':
            recommendations = [
                "• <b>Regular check-ups</b> - Schedule routine visits with healthcare provider",
                "• <b>Heart-healthy diet</b> - Focus on fruits, vegetables, and lean proteins",
                "• <b>Regular exercise</b> - Aim for 150 minutes moderate exercise weekly",
                "• <b>Risk monitoring</b> - Track blood pressure and cholesterol levels"
            ]
        else:
            recommendations = [
                "• <b>Maintain healthy lifestyle</b> - Continue current healthy habits",
                "• <b>Balanced nutrition</b> - Eat variety of nutritious foods, limit processed foods",
                "• <b>Stay active</b> - Maintain regular physical activity routine",
                "• <b>Preventive care</b> - Keep up with routine health screenings"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, self.bullet_style))
        
        story.append(Spacer(1, 0.12*inch))
        
        # Model Information (more compact)
        story.append(Paragraph("Assessment Methodology", self.subsection_heading_style))
        model_info = ("This assessment employs three advanced AI models (K-Nearest Neighbors, Random Forest, and XGBoost) "
                     "trained on clinical data to provide consensus predictions from 13 health parameters.")
        story.append(Paragraph(model_info, self.body_style))
        
        story.append(Spacer(1, 0.08*inch))
        
        # Disclaimer
        disclaimer_text = ("<b>Medical Disclaimer:</b> This report is for informational purposes only and does not constitute medical advice. "
                          "Always consult qualified healthcare providers before making health decisions. "
                          "This AI assessment supplements, but does not replace, professional medical evaluation.")
        story.append(Paragraph(disclaimer_text, self.disclaimer_style))
        
        return story

def generate_pdf_report(explanation, input_data, feature_descriptions, session_id):
    """Generate a professional PDF report for heart disease prediction"""
    try:
        # Create the PDF generator
        pdf_gen = ProfessionalPDFReport()
        
        # Create buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=60,
            leftMargin=60,
            topMargin=60,
            bottomMargin=60
        )
        
        # Build the story with proper page structure
        story = []
        
        # Page 1: Header, Risk Level, Patient Information
        story.extend(pdf_gen.create_page_1(explanation, input_data, feature_descriptions, session_id))
        story.append(PageBreak())
        
        # Page 2: Visualizations
        story.extend(pdf_gen.create_page_2(explanation))
        story.append(PageBreak())
        
        # Page 3: Comparisons and Recommendations
        story.extend(pdf_gen.create_page_3(explanation, input_data))

        # Build the PDF
        doc.build(story)
        pdf_value = buffer.getvalue()
        buffer.close()
        
        return pdf_value
        
    except Exception as e:
        print(f"Error generating professional PDF report: {str(e)}")
        raise

# Legacy function wrappers for backward compatibility
def create_header(patient_id, risk_level, probability):
    """Legacy function - maintained for compatibility"""
    pdf_gen = ProfessionalPDFReport()
    return pdf_gen.create_page_1({'risk_level': risk_level, 'probability': probability}, {}, {}, patient_id)

def create_patient_data_section(input_data, feature_descriptions):
    """Legacy function - maintained for compatibility"""
    return []

def create_risk_factors_section(top_risk_factors, feature_descriptions):
    """Legacy function - maintained for compatibility"""
    return []

def create_recommendation_section(risk_level, top_risk_factors=None):
    """Legacy function - maintained for compatibility"""
    return []

def embed_image(base64_img, width=6*inch, height=4*inch):
    """Convert a base64 image to a ReportLab Image object"""
    try:
        img_data = base64.b64decode(base64_img)
        img_temp = BytesIO(img_data)
        img = Image(img_temp, width=width, height=height)
        return img
    except Exception as e:
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        return Paragraph(f"Error processing image: {str(e)}", styles['Normal'])