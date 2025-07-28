import os
import json
import re
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiHeartDoctorChatbot:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.conversation_stage = "initial_greeting"
        self.conversation_context = []
        self.user_symptoms = {}
        self.detected_conditions = set()
        self.severity_level = "unknown"
        self.exchange_count = 0  # Track number of exchanges
        self.consultation_complete = False
        
        # For compatibility with existing app.py code
        self.current_stage = "initial_greeting"
        self.consultation_data = {}
        self.symptom_details = {}
        
        # Emergency keywords that bypass AI for immediate response
        # Note: Symptoms like chest pain, breathing issues are now checked for severity
        self.emergency_keywords = [
            'heart attack', 'emergency', '1122', 'help', 'dying',
            'unconscious', 'collapse', 'heart stopped'
        ]
        
        # Urgent symptoms that need rapid assessment (not full emergency but urgent)
        self.urgent_keywords = [
            'not able to breath', 'unable to breath', 'difficulty breathing',
            'shortness of breath', 'heart pain', 'pain in my heart', 
            'chest discomfort', 'heart racing', 'palpitations',
            'pain on my left side', 'left chest pain', 'choking'
        ]
        
        # Medical conversation stages with proper progression
        self.stages = {
            "initial_greeting": "Welcome and initial assessment",
            "chief_complaint": "Understanding primary concern",
            "symptom_analysis": "Detailed symptom evaluation",
            "symptom_details": "Specific symptom characteristics",
            "medical_history": "Past medical history review",
            "risk_factors": "Cardiovascular risk assessment",
            "assessment_summary": "Clinical summary",
            "recommendations": "Treatment recommendations",
            "completed": "Consultation completed"
        }
        
        # Enhanced symptom patterns with natural language severity
        self.symptom_patterns = {
            'chest_pain': {
                'keywords': ['chest pain', 'chest discomfort', 'chest pressure', 'chest tightness', 
                           'heart pain', 'breast pain', 'thoracic pain'],
                'patterns': [r'pain.*chest', r'chest.*pain', r'pain.*left.*chest', r'chest.*side.*pain',
                           r'heart.*hurt', r'chest.*tight', r'pressure.*chest'],
                'severity_mild': ['mild', 'slight', 'little', 'minor', 'light', 'bearable'],
                'severity_moderate': ['moderate', 'noticeable', 'uncomfortable', 'concerning', 'disturbing'],
                'severity_severe': ['severe', 'intense', 'excruciating', 'unbearable', 'crushing', 'sharp', 'terrible']
            },
            'dyspnea': {
                'keywords': ['shortness of breath', 'difficulty breathing', 'breathless', 
                           'can\'t breathe', 'breathing problems', 'out of breath', 'choking'],
                'patterns': [r'short.*breath', r'breath.*short', r'hard.*breath', r'difficult.*breath', r'not.*able.*breath'],
                'severity_mild': ['mild', 'slight', 'walking fast', 'little difficulty'],
                'severity_moderate': ['stairs', 'walking', 'talking', 'noticeable', 'concerning'],
                'severity_severe': ['rest', 'sitting', 'lying down', 'severe', 'very difficult', 'choking']
            },
            'palpitations': {
                'keywords': ['palpitations', 'heart racing', 'rapid heartbeat', 'heart pounding',
                           'irregular heartbeat', 'heart skipping'],
                'patterns': [r'heart.*racing', r'heart.*fast', r'heart.*pound', r'heart.*skip'],
                'severity_mild': ['occasional', 'mild', 'slight', 'sometimes'],
                'severity_moderate': ['frequent', 'noticeable', 'concerning', 'regular'],
                'severity_severe': ['constant', 'severe', 'very fast', 'non-stop']
            },
            'syncope': {
                'keywords': ['dizziness', 'lightheaded', 'fainting', 'syncope', 'dizzy spells'],
                'patterns': [r'feel.*dizzy', r'light.*head', r'pass.*out', r'faint'],
                'severity_mild': ['mild', 'slight', 'occasional', 'little'],
                'severity_moderate': ['frequent', 'noticeable', 'concerning'],
                'severity_severe': ['severe', 'fainting', 'unconscious', 'constant']
            },
            'fatigue': {
                'keywords': ['fatigue', 'tired', 'exhausted', 'weak', 'no energy'],
                'patterns': [r'feel.*tired', r'very.*tired', r'no.*energy', r'weak'],
                'severity_mild': ['mild', 'slight', 'little tired', 'somewhat'],
                'severity_moderate': ['noticeable', 'more tired than usual', 'concerning'],
                'severity_severe': ['severe', 'exhausted', 'can\'t function', 'very weak']
            },
            'edema': {
                'keywords': ['swelling', 'edema', 'swollen legs', 'swollen feet', 'fluid retention'],
                'patterns': [r'swoll.*leg', r'swoll.*feet', r'swoll.*ankle', r'fluid.*retain'],
                'severity_mild': ['mild', 'slight', 'minimal', 'little'],
                'severity_moderate': ['noticeable', 'moderate', 'concerning'],
                'severity_severe': ['severe', 'significant', 'marked', 'very swollen']
            }
        }

    def configure_api(self, api_key: str) -> bool:
        """Configure Google Gemini API with the provided key."""
        try:
            self.api_key = api_key
            genai.configure(api_key=api_key)
            
            # Initialize the model with safety settings
            self.model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',  # Using the latest free model
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            # Test the API with a simple request
            test_response = self.model.generate_content("Hello, please respond with 'API configured successfully'")
            
            if test_response and test_response.text:
                return True
            return False
            
        except Exception as e:
            print(f"Gemini API configuration error: {e}")
            return False

    def detect_emergency(self, message: str) -> bool:
        """Detect emergency situations from user message."""
        message_lower = message.lower()
        
        # First check for explicit emergency keywords (these are always emergency regardless of severity)
        explicit_emergency_keywords = [
            'heart attack', 'emergency', '1122', 'help', 'dying',
            'unconscious', 'collapse', 'heart stopped'
        ]
        if any(keyword in message_lower for keyword in explicit_emergency_keywords):
            return True
        
        # Analyze all symptoms for severity-based emergency detection
        symptoms = self.analyze_symptoms_flexible(message)
        
        # Check each detected symptom for emergency severity
        for symptom_name, symptom_data in symptoms.items():
            severity = symptom_data['severity']
            
            # Only trigger emergency for severe symptoms
            if severity == 'severe':
                # Check if this symptom type can be emergency
                if symptom_name in ['chest_pain', 'dyspnea', 'syncope', 'palpitations']:
                    return True
        
        # Check for specific severe pain mentions
        if 'severe pain' in message_lower or 'crushing pain' in message_lower:
            return True
        
        return False

    def detect_urgent(self, message: str) -> bool:
        """Detect urgent (but not emergency) situations that need rapid assessment."""
        message_lower = message.lower()
        
        # Check for urgent keywords that don't need severity analysis
        urgent_keywords_no_severity = [
            'not able to breath', 'unable to breath', 'difficulty breathing',
            'shortness of breath', 'choking'
        ]
        if any(keyword in message_lower for keyword in urgent_keywords_no_severity):
            return True
        
        # Analyze all symptoms for severity-based urgent detection
        symptoms = self.analyze_symptoms_flexible(message)
        
        # Check each detected symptom for urgent severity
        for symptom_name, symptom_data in symptoms.items():
            severity = symptom_data['severity']
            
            # Consider moderate symptoms as urgent
            if severity == 'moderate':
                # Check if this symptom type can be urgent
                if symptom_name in ['chest_pain', 'dyspnea', 'palpitations', 'syncope']:
                    return True
            
            # Consider severe symptoms as urgent (but not emergency - they'll be caught by emergency detection)
            elif severity == 'severe':
                # Check if this symptom type can be urgent
                if symptom_name in ['chest_pain', 'dyspnea', 'palpitations', 'syncope']:
                    return True
        
        return False

    def analyze_symptoms_flexible(self, message: str) -> Dict:
        """Analyze symptoms using flexible pattern matching."""
        message_lower = message.lower()
        detected_symptoms = {}
        overall_severity = "mild"
        
        for symptom_name, symptom_data in self.symptom_patterns.items():
            detected = False
            severity = "mild"
            matched_phrase = ""
            
            # Check direct keywords
            for keyword in symptom_data['keywords']:
                if keyword in message_lower:
                    detected = True
                    matched_phrase = keyword
                    break
            
            # Check regex patterns
            if not detected:
                for pattern in symptom_data['patterns']:
                    if re.search(pattern, message_lower):
                        detected = True
                        match = re.search(pattern, message_lower)
                        matched_phrase = match.group() if match else pattern
                        break
            
            if detected:
                # Determine severity
                for sev_keyword in symptom_data['severity_severe']:
                    if sev_keyword in message_lower:
                        severity = "severe"
                        overall_severity = "severe"
                        break
                
                if severity != "severe":
                    for sev_keyword in symptom_data['severity_moderate']:
                        if sev_keyword in message_lower:
                            severity = "moderate"
                            if overall_severity != "severe":
                                overall_severity = "moderate"
                            break
                
                detected_symptoms[symptom_name] = {
                    'severity': severity,
                    'matched_phrase': matched_phrase
                }
                self.detected_conditions.add(symptom_name)
        
        self.severity_level = overall_severity
        return detected_symptoms

    def validate_medical_input(self, user_message: str) -> bool:
        """Validate if the input is related to medical/cardiovascular topics."""
        user_message_lower = user_message.lower()
        
        # Clear non-medical topics that should be rejected
        non_medical_patterns = [
            'weather', 'joke', 'funny', 'cook', 'recipe', 'math', 'calculate',
            'music', 'movie', 'book', 'game', 'sport', 'politics', 'news',
            'shopping', 'restaurant', 'travel', 'vacation', 'work', 'job',
            'school', 'homework', 'computer', 'software', 'programming',
            '2+2', '1+1', 'what is', 'what\'s', 'how much', 'tell me', 'story'
        ]
        
        # First check for obvious non-medical content
        for pattern in non_medical_patterns:
            if pattern in user_message_lower:
                return False
        
        # Specific medical/cardiovascular keywords
        medical_keywords = [
            'pain', 'chest', 'heart', 'shortness', 'breath', 'dizzy', 'palpitations',
            'fatigue', 'tired', 'swelling', 'pressure', 'discomfort', 'ache',
            'symptoms', 'sick', 'hurt', 'medication', 'blood pressure', 'attack', 
            'emergency', 'hospital', 'breathing', 'pulse', 'rhythm', 'irregular',
            'nausea', 'vomiting', 'sweating', 'fever', 'temperature', 'exercise',
            'walking', 'stairs', 'activity', 'doctor', 'medical', 'health'
        ]
        
        # Check if message contains medical keywords
        for keyword in medical_keywords:
            if keyword in user_message_lower:
                return True
        
        # Check if it's a greeting or medical conversation response
        medical_responses = [
            'hello', 'hi', 'hey', 'good', 'thank', 'thanks', 'yes', 'no', 'ok', 'okay',
            'started', 'began', 'this morning', 'yesterday', 'last week', 'feels', 'feeling',
            'when', 'where', 'how', 'what', 'why', 'since', 'for', 'ago'
        ]
        
        # Allow these only if they seem like medical conversation responses
        for response in medical_responses:
            if response in user_message_lower:
                return True
            
        return False

    def update_conversation_stage(self):
        """Update conversation stage based on exchange count and content."""
        if self.consultation_complete:
            self.conversation_stage = "completed"
            self.current_stage = "completed"
            return
        
        # Progressive stage advancement based on exchange count
        if self.exchange_count <= 1:
            self.conversation_stage = "initial_greeting"
        elif self.exchange_count == 2:
            self.conversation_stage = "chief_complaint"
        elif self.exchange_count == 3:
            self.conversation_stage = "symptom_analysis"
        elif self.exchange_count == 4:
            self.conversation_stage = "medical_history"
        elif self.exchange_count == 5:
            self.conversation_stage = "assessment_summary"
        elif self.exchange_count >= 6:
            self.conversation_stage = "recommendations"
            # Mark as complete after recommendations
            if self.exchange_count >= 7:
                self.consultation_complete = True
                self.conversation_stage = "completed"
        
        # Update compatibility property
        self.current_stage = self.conversation_stage

    def build_medical_prompt(self, user_message: str) -> str:
        """Build a comprehensive medical consultation prompt for Gemini."""
        
        # Validate input is medical
        if not self.validate_medical_input(user_message):
            return "NON_MEDICAL_INPUT"
        
        # Detect symptoms in current message
        current_symptoms = self.analyze_symptoms_flexible(user_message)
        
        # Update conversation stage
        self.update_conversation_stage()
        
        system_prompt = """You are a professional cardiovascular specialist conducting a medical consultation. You have extensive experience in cardiology and emergency medicine.

CRITICAL MEDICAL GUIDELINES:
- ONLY discuss cardiovascular and related medical topics
- NEVER use casual greetings like "Good morning/afternoon" after initial greeting
- NEVER identify yourself as "Dr. Gemini" or any specific name
- NEVER repeat the initial greeting or start over - maintain conversation continuity
- PRIORITIZE SPEED for patients in distress - don't waste time with slow questioning
- If patient has breathing issues, chest pain, or heart symptoms - ASSESS URGENCY IMMEDIATELY
- Ask multiple relevant questions together to get critical information quickly
- Provide immediate guidance for concerning symptoms
- Focus on rapid symptom assessment for patient safety
- Use natural language severity descriptions (mild, moderate, severe) instead of numerical scales
- Use Pakistan emergency number 1122 instead of 911
- ALWAYS continue the conversation naturally without restarting or repeating greetings

CONSULTATION PROTOCOL:
1. INITIAL GREETING (Exchange 1): Welcome and ask for chief complaint
2. CHIEF COMPLAINT (Exchange 2): Understand primary concern and symptoms
3. SYMPTOM ANALYSIS (Exchange 3): Detailed symptom evaluation and characteristics
4. MEDICAL HISTORY (Exchange 4): Past medical history and risk factors
5. ASSESSMENT SUMMARY (Exchange 5): Clinical summary and initial assessment
6. RECOMMENDATIONS (Exchange 6): Treatment recommendations and next steps
7. COMPLETION (Exchange 7+): Final guidance and conclusion

RESPONSE REQUIREMENTS:
- If patient mentions: "can't breathe", "chest pain", "heart pain" → RAPID ASSESSMENT MODE
- Ask 3-4 critical questions together instead of one at a time
- Include immediate safety advice when appropriate
- Be efficient - patients in distress need quick help, not lengthy questioning
- Provide clear action steps (call 1122, seek immediate care, etc.)
- Balance thoroughness with speed - get essential info fast
- Use natural severity descriptions: mild, moderate, severe
- NEVER repeat greetings or start over - maintain conversation continuity

CURRENT CONSULTATION STATUS:
"""

        # Add current stage and exchange count
        system_prompt += f"Current Stage: {self.stages.get(self.conversation_stage, 'General consultation')}\n"
        system_prompt += f"Exchange Number: {self.exchange_count}\n"
        
        # Add detected symptoms
        if current_symptoms:
            system_prompt += f"Recently Detected Symptoms: {', '.join([f'{k} ({v["severity"]})' for k, v in current_symptoms.items()])}\n"
        
        if self.user_symptoms:
            system_prompt += f"Previously Discussed Symptoms: {', '.join([f'{k} ({v.get("severity", "unknown")})' for k, v in self.user_symptoms.items()])}\n"
        
        # Add conversation history context (limited to last 3 exchanges)
        if self.conversation_context:
            system_prompt += f"\nRecent conversation context:\n"
            for entry in self.conversation_context[-3:]:  # Last 3 exchanges only
                system_prompt += f"Patient: {entry.get('user', '')}\nDoctor: {entry.get('assistant', '')}\n"
        
        # Detect urgency level
        is_urgent = self.detect_urgent(user_message)
        has_critical_symptoms = any(symptom in ['chest_pain', 'dyspnea'] for symptom in current_symptoms.keys())
        
        if is_urgent or has_critical_symptoms:
            # Analyze severity of all detected symptoms
            mild_symptoms = []
            moderate_symptoms = []
            severe_symptoms = []
            
            for symptom_name, symptom_data in current_symptoms.items():
                severity = symptom_data['severity']
                if severity == 'mild':
                    mild_symptoms.append(symptom_name)
                elif severity == 'moderate':
                    moderate_symptoms.append(symptom_name)
                elif severity == 'severe':
                    severe_symptoms.append(symptom_name)
            
            # Handle based on severity levels
            if severe_symptoms:
                system_prompt += f"""
🚨 SEVERE SYMPTOMS DETECTED - EMERGENCY ASSESSMENT MODE
Patient has severe symptoms: {', '.join(severe_symptoms)} which require immediate medical attention.

EMERGENCY RESPONSE REQUIREMENTS:
- Assess urgency immediately
- Ask critical questions about onset, severity, and associated symptoms
- Provide immediate safety guidance
- Recommend emergency care if appropriate
- Be efficient and direct

Patient's current message: "{user_message}"

Provide emergency assessment and immediate guidance:"""
            elif moderate_symptoms:
                system_prompt += f"""
⚠️ MODERATE SYMPTOMS DETECTED - URGENT ASSESSMENT MODE
Patient has moderate symptoms: {', '.join(moderate_symptoms)} which require prompt evaluation.

URGENT RESPONSE REQUIREMENTS:
- Assess symptoms thoroughly but efficiently
- Ask about onset, duration, and characteristics
- Provide appropriate urgency guidance
- Recommend timely medical evaluation

Patient's current message: "{user_message}"

Provide urgent assessment and guidance:"""
            elif mild_symptoms:
                system_prompt += f"""
MILD SYMPTOMS DETECTED - STANDARD ASSESSMENT MODE
Patient has mild symptoms: {', '.join(mild_symptoms)} which require careful evaluation but are not immediately life-threatening.

MILD SYMPTOMS RESPONSE REQUIREMENTS:
- Acknowledge the symptoms but remain calm and professional
- Ask about onset, duration, and characteristics
- Inquire about triggers and relieving factors
- Assess for associated symptoms
- Provide appropriate guidance for mild symptoms
- No need for emergency language or immediate action

Patient's current message: "{user_message}"

Provide professional assessment for mild symptoms:"""
            else:
                system_prompt += f"""
🚨 URGENT CASE DETECTED - RAPID ASSESSMENT MODE ACTIVATED
Patient has urgent cardiovascular symptoms requiring immediate and efficient evaluation.

URGENT RESPONSE REQUIREMENTS - RAPID MODE:
- Patient has urgent symptoms: breathing difficulty, chest/heart pain
- Ask 3-4 critical questions together: onset time, severity description (mild/moderate/severe), associated symptoms, current location
- Include immediate safety guidance (when to call 1122, seek emergency care)
- Be efficient but thorough - get essential information quickly
- Provide clear action steps based on symptom severity
- No time for lengthy single-question exchanges
- Use natural language: "How severe is your pain - mild, moderate, or severe?"

Patient's current message: "{user_message}"

Provide RAPID cardiovascular assessment with multiple questions and immediate guidance:"""
        else:
            # Stage-specific prompts
            if self.conversation_stage == "initial_greeting":
                system_prompt += f"""
INITIAL GREETING RESPONSE:
- Welcome the patient professionally (ONLY on first exchange)
- Ask for their chief complaint or main concern
- Encourage them to describe any symptoms they're experiencing
- Set the tone for a professional medical consultation

Patient's current message: "{user_message}"

Provide professional initial greeting and ask for chief complaint:"""
            
            elif self.conversation_stage == "chief_complaint":
                system_prompt += f"""
CHIEF COMPLAINT ASSESSMENT:
- DO NOT use any greetings - continue conversation naturally
- Focus on understanding the primary concern
- Ask about symptom onset, duration, and characteristics
- Assess urgency level of the complaint
- Gather essential symptom details

Patient's current message: "{user_message}"

Assess the chief complaint and gather symptom details:"""
            
            elif self.conversation_stage == "symptom_analysis":
                system_prompt += f"""
SYMPTOM ANALYSIS:
- DO NOT use any greetings - continue conversation naturally
- Conduct detailed symptom evaluation
- Ask about symptom patterns, triggers, and relieving factors
- Assess severity using natural language (mild, moderate, severe)
- Gather comprehensive symptom history

Patient's current message: "{user_message}"

Conduct detailed symptom analysis:"""
            
            elif self.conversation_stage == "medical_history":
                system_prompt += f"""
MEDICAL HISTORY REVIEW:
- DO NOT use any greetings - continue conversation naturally
- Ask about personal and family medical history
- Inquire about current medications and allergies
- Assess cardiovascular risk factors
- Gather relevant medical background

Patient's current message: "{user_message}"

Review medical history and risk factors:"""
            
            elif self.conversation_stage == "assessment_summary":
                system_prompt += f"""
ASSESSMENT SUMMARY:
- DO NOT use any greetings - continue conversation naturally
- Provide clinical summary of findings
- Assess overall risk level
- Identify key concerns and priorities
- Prepare for recommendations

Patient's current message: "{user_message}"

Provide clinical assessment summary:"""
            
            elif self.conversation_stage == "recommendations":
                system_prompt += f"""
TREATMENT RECOMMENDATIONS:
- DO NOT use any greetings - continue conversation naturally
- Provide specific medical recommendations
- Give clear next steps and action items
- Include emergency guidance if needed
- Offer follow-up advice

Patient's current message: "{user_message}"

Provide treatment recommendations and next steps:"""
            
            elif self.conversation_stage == "completed":
                system_prompt += f"""
CONSULTATION COMPLETION:
- DO NOT use any greetings - continue conversation naturally
- Summarize the consultation
- Provide final recommendations
- Give clear next steps
- Thank the patient professionally

Patient's current message: "{user_message}"

Provide final consultation summary and conclusion:"""
            
            else:
                system_prompt += f"""
STANDARD RESPONSE REQUIREMENTS:
- Ask relevant medical questions to assess cardiovascular symptoms
- Be professional and efficient
- Provide appropriate medical guidance
- Keep responses focused and informative
- Use natural severity descriptions instead of numerical scales

Patient's current message: "{user_message}"

Provide your cardiovascular assessment response:"""

        return system_prompt

    def get_ai_response(self, user_message: str) -> str:
        """Get AI-powered response from Google Gemini."""
        if not self.model:
            return self.get_fallback_response(user_message)
        
        try:
            prompt = self.build_medical_prompt(user_message)
            
            # Handle non-medical input
            if prompt == "NON_MEDICAL_INPUT":
                return self.get_non_medical_response()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent medical responses
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=400,  # Slightly longer for comprehensive responses
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return self.get_fallback_response(user_message)
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self.get_fallback_response(user_message)

    def get_non_medical_response(self) -> str:
        """Response for non-medical inputs."""
        return """I'm here to conduct a cardiovascular health assessment. Please share any symptoms or concerns related to your heart health, chest pain, breathing difficulties, or other cardiovascular symptoms you're experiencing.

For medical emergencies, please call emergency services immediately."""

    def get_fallback_response(self, user_message: str) -> str:
        """Professional fallback responses when AI is unavailable."""
        # Check for non-medical input first
        if not self.validate_medical_input(user_message):
            return self.get_non_medical_response()
        
        # Check for urgent symptoms first
        is_urgent = self.detect_urgent(user_message)
        symptoms = self.analyze_symptoms_flexible(user_message)
        has_critical_symptoms = any(symptom in ['chest_pain', 'dyspnea'] for symptom in symptoms.keys())
        
        # Rapid assessment for urgent cases
        if is_urgent or has_critical_symptoms:
            if symptoms:
                # Analyze severity of all symptoms
                mild_symptoms = []
                moderate_symptoms = []
                severe_symptoms = []
                
                for symptom_name, symptom_data in symptoms.items():
                    severity = symptom_data['severity']
                    if severity == 'mild':
                        mild_symptoms.append(symptom_name)
                    elif severity == 'moderate':
                        moderate_symptoms.append(symptom_name)
                    elif severity == 'severe':
                        severe_symptoms.append(symptom_name)
                
                # Handle based on severity levels
                if severe_symptoms:
                    symptom_display = ', '.join([s.replace('_', ' ') for s in severe_symptoms])
                    return f"""You've described severe {symptom_display} which requires immediate medical attention. I need critical information quickly:

1. When did this start? (minutes/hours/days ago)
2. How severe is it - mild, moderate, or severe?
3. Any associated symptoms like chest pressure, sweating, nausea, or arm pain?
4. Are you currently at home, work, or can you get to a hospital?

If symptoms are severe or worsening - call 1122 immediately.
Please answer these questions so I can guide you appropriately."""
                
                elif moderate_symptoms:
                    symptom_display = ', '.join([s.replace('_', ' ') for s in moderate_symptoms])
                    return f"""You've described moderate {symptom_display} which requires prompt evaluation. Let me gather some important information:

1. When did this start? (minutes/hours/days ago)
2. What makes it better or worse?
3. Any associated symptoms?
4. Are you able to get to medical care if needed?

This will help me provide appropriate guidance for your symptoms."""
                
                elif mild_symptoms:
                    symptom_display = ', '.join([s.replace('_', ' ') for s in mild_symptoms])
                    return f"""You've described mild {symptom_display}. Let me gather some important information to assess this properly:

1. When did this start? (minutes/hours/days ago)
2. What does it feel like?
3. What makes it better or worse?
4. Any associated symptoms?

This will help me provide appropriate guidance for your symptoms."""
                
                else:
                    symptom_names = list(symptoms.keys())
                    symptom_display = symptom_names[0].replace('_', ' ')
                    return f"""You've described {symptom_display} which requires immediate assessment. I need critical information quickly:

1. When did this start? (minutes/hours/days ago)
2. How severe is it - mild, moderate, or severe?
3. Any associated symptoms?
4. Are you currently at home, work, or can you get to a hospital?

Please answer these questions so I can guide you appropriately."""
            else:
                return """You're experiencing concerning symptoms that need rapid evaluation. Please tell me:

1. When did your breathing difficulty/chest pain start?
2. How severe is it - mild, moderate, or severe?
3. Any chest pressure, sweating, nausea, or arm pain?
4. Are you able to get to emergency care if needed?

If severe or worsening - call 1122 now.
Otherwise, seek urgent medical care immediately."""
        
        # Stage-based medical responses
        stage_responses = {
            "initial_greeting": "Please describe any cardiovascular symptoms or concerns you're experiencing, such as chest pain, shortness of breath, palpitations, or other heart-related symptoms.",
            "chief_complaint": "Please provide more details about when these symptoms started and their characteristics.",
            "symptom_analysis": "Describe when symptoms started, their severity (mild, moderate, or severe), and any triggers or patterns you've noticed.",
            "medical_history": "Do you have any personal or family history of cardiovascular disease, high blood pressure, diabetes, or other relevant medical conditions?",
            "assessment_summary": "Based on the symptoms you've described, I need to gather a bit more information about your medical history and risk factors.",
            "recommendations": "Based on the symptoms you've described, I recommend evaluation by a healthcare provider for proper cardiovascular assessment.",
            "completed": "Thank you for the comprehensive information. Please consult with a healthcare professional for proper medical evaluation and follow-up care."
        }
        
        return stage_responses.get(self.conversation_stage, 
                                 "Please describe your current cardiovascular symptoms including when they started and their severity level.")

    def process_message(self, user_message: str) -> Tuple[str, bool]:
        """Process user message and return response with emergency flag."""
        user_message = user_message.strip()
        
        # Check for emergency first
        if self.detect_emergency(user_message):
            emergency_response = """🚨 **EMERGENCY SITUATION DETECTED** 🚨

Based on your symptoms, this could be a medical emergency. Please:

**IMMEDIATELY:**
- Call 1122 or go to the nearest emergency room
- If experiencing chest pain, chew an aspirin if not allergic
- Do not drive yourself - call an ambulance or have someone drive you

**DO NOT DELAY** - Heart emergencies require immediate medical attention.

While waiting for help:
- Sit down and try to stay calm
- Loosen tight clothing
- If you lose consciousness, someone should call 1122 immediately

Your symptoms require immediate professional medical evaluation. Please seek emergency care now."""
            
            return emergency_response, True
        
        # Increment exchange count
        self.exchange_count += 1
        
        # Get AI response for normal consultation
        response = self.get_ai_response(user_message)
        
        # Update conversation context
        self.conversation_context.append({
            'user': user_message,
            'assistant': response
        })
        
        # Keep only recent context to manage memory
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        # Update user symptoms
        current_symptoms = self.analyze_symptoms_flexible(user_message)
        self.user_symptoms.update(current_symptoms)
        
        # Update conversation stage
        self.update_conversation_stage()
        
        return response, False

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation for reports."""
        return {
            'stage': self.conversation_stage,
            'detected_symptoms': dict(self.user_symptoms),
            'severity_level': self.severity_level,
            'conversation_length': len(self.conversation_context),
            'conditions_discussed': list(self.detected_conditions),
            'exchange_count': self.exchange_count,
            'completed': self.consultation_complete
        }

    def reset_conversation(self):
        """Reset the conversation state."""
        self.conversation_stage = "initial_greeting"
        self.conversation_context = []
        self.user_symptoms = {}
        self.detected_conditions = set()
        self.severity_level = "unknown"
        self.exchange_count = 0
        self.consultation_complete = False
        # Reset compatibility properties
        self.current_stage = "initial_greeting"
        self.consultation_data = {}
        self.symptom_details = {}

    # Compatibility methods for existing app.py code
    def get_initial_greeting(self):
        """Get initial greeting message."""
        return "I'm here to conduct a cardiovascular health assessment. Please describe your symptoms including: what you're experiencing, when it started, and describe severity as mild, moderate, or severe. For urgent symptoms (chest pain, breathing difficulty), I'll assess immediately."

    def process_response(self, user_message: str) -> Tuple[str, Dict]:
        """Process user response and return next question and data."""
        response, is_emergency = self.process_message(user_message)
        
        # Update compatibility properties
        self.current_stage = self.conversation_stage
        self.consultation_data.update(self.user_symptoms)
        self.symptom_details.update(self.user_symptoms)
        
        # Return format expected by app.py
        completed_data = None
        if self.consultation_complete or self.exchange_count >= 7:
            self.current_stage = "completed"
            completed_data = {
                'symptoms': self.user_symptoms,
                'severity': self.severity_level,
                'conditions': list(self.detected_conditions),
                'emergency': is_emergency,
                'exchange_count': self.exchange_count
            }
        
        return response, completed_data

    def get_current_question(self) -> Dict:
        """Get current question in expected format."""
        if self.current_stage == "completed":
            return {'message': "Assessment complete. Please consult with a healthcare professional for proper medical evaluation and follow-up care."}
        
        stage_questions = {
            "initial_greeting": "Please describe any cardiovascular symptoms or concerns you're experiencing, such as chest pain, shortness of breath, palpitations, or other heart-related symptoms.",
            "chief_complaint": "Please provide more details about when these symptoms started and their characteristics.",
            "symptom_analysis": "Describe when symptoms started, their severity (mild, moderate, or severe), and any triggers or patterns you've noticed.",
            "medical_history": "Do you have any personal or family history of cardiovascular disease, high blood pressure, diabetes, or other relevant medical conditions?",
            "assessment_summary": "Based on the symptoms you've described, I need to gather a bit more information about your medical history and risk factors.",
            "recommendations": "Based on the symptoms you've described, I recommend evaluation by a healthcare provider for proper cardiovascular assessment."
        }
        
        return {'message': stage_questions.get(self.current_stage, "Please describe your current cardiovascular symptoms in more detail.")}

    def reset(self):
        """Reset chatbot state - compatibility method."""
        self.reset_conversation()
    
    def test_severity_detection(self, message: str) -> Dict:
        """Test method to analyze symptom severity detection."""
        symptoms = self.analyze_symptoms_flexible(message)
        is_emergency = self.detect_emergency(message)
        is_urgent = self.detect_urgent(message)
        
        return {
            'symptoms': symptoms,
            'is_emergency': is_emergency,
            'is_urgent': is_urgent,
            'message': message
        } 