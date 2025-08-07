from .heart_disease_info import HeartDiseaseInfo
import re

class ProfessionalHeartDoctorChatbot:
    def __init__(self):
        self.consultation_data = {}
        self.current_stage = 'initial_greeting'
        self.follow_up_questions = []
        self.symptom_details = {}
        self.heart_disease_info = HeartDiseaseInfo()
        
        # Professional consultation stages
        self.consultation_stages = {
            'initial_greeting': self._handle_initial_greeting,
            'chief_complaint': self._handle_chief_complaint,
            'symptom_analysis': self._handle_symptom_analysis,
            'symptom_details': self._handle_symptom_details,
            'medical_history': self._handle_medical_history,
            'risk_factors': self._handle_risk_factors,
            'assessment_summary': self._handle_assessment_summary,
            'recommendations': self._handle_recommendations
        }
        
        # Enhanced symptom patterns with flexible matching
        self.symptom_patterns = {
            'chest_pain': {
                'keywords': [
                    'chest pain', 'chest ache', 'chest discomfort', 'chest pressure', 'chest tightness',
                    'crushing pain', 'burning sensation', 'sharp chest pain', 'chest hurt', 'chest burning'
                ],
                'flexible_patterns': [
                    r'pain.*chest', r'chest.*pain', r'pain.*left.*chest', r'pain.*right.*chest',
                    r'hurt.*chest', r'chest.*hurt', r'ache.*chest', r'chest.*ache',
                    r'pressure.*chest', r'chest.*pressure', r'discomfort.*chest', r'chest.*discomfort',
                    r'pain.*side.*chest', r'chest.*side.*pain', r'left.*side.*chest.*pain',
                    r'right.*side.*chest.*pain', r'pain.*in.*chest', r'chest.*area.*pain'
                ],
                'severity_keywords': {
                    'severe': ['crushing', 'severe', 'excruciating', 'unbearable', 'intense', 'sharp stabbing', 'tearing', 'extreme'],
                    'moderate': ['moderate', 'aching', 'pressure', 'tight', 'heavy', 'squeezing', 'noticeable'],
                    'mild': ['mild', 'slight', 'minor', 'dull', 'occasional', 'intermittent', 'little', 'small']
                },
                'base_severity': 'high',
                'follow_ups': ['location', 'quality', 'radiation', 'duration', 'triggers', 'relieving_factors']
            },
            'dyspnea': {
                'keywords': [
                    'shortness of breath', 'difficulty breathing', 'breathless', 'can\'t breathe', 
                    'breathlessness', 'short of breath', 'gasping', 'trouble breathing'
                ],
                'flexible_patterns': [
                    r'short.*breath', r'breath.*short', r'difficult.*breath', r'trouble.*breath',
                    r'hard.*breath', r'can.*breath', r'cannot.*breath', r'gasping'
                ],
                'severity_keywords': {
                    'severe': ['can\'t breathe', 'gasping', 'suffocating', 'choking', 'cannot breathe'],
                    'moderate': ['difficult breathing', 'breathless', 'winded', 'trouble breathing'],
                    'mild': ['slightly breathless', 'mild shortness', 'little breathless']
                },
                'base_severity': 'high',
                'follow_ups': ['onset', 'exertion_related', 'position_related', 'duration']
            },
            'palpitations': {
                'keywords': [
                    'palpitations', 'heart racing', 'heart pounding', 'irregular heartbeat', 
                    'heart flutter', 'rapid heartbeat', 'fast heartbeat'
                ],
                'flexible_patterns': [
                    r'heart.*racing', r'heart.*fast', r'heart.*pounding', r'heart.*beating',
                    r'fast.*heart', r'rapid.*heart', r'irregular.*heart', r'heart.*flutter'
                ],
                'severity_keywords': {
                    'severe': ['pounding', 'racing', 'very fast', 'irregular', 'extremely fast'],
                    'moderate': ['noticeable', 'flutter', 'skipping', 'fast'],
                    'mild': ['occasional', 'slight flutter', 'aware of heartbeat', 'little fast']
                },
                'base_severity': 'medium',
                'follow_ups': ['frequency', 'duration', 'triggers', 'associated_symptoms']
            },
            'syncope': {
                'keywords': [
                    'fainting', 'fainted', 'lost consciousness', 'passing out', 'blackout', 
                    'dizzy spells', 'lightheaded', 'feeling faint'
                ],
                'flexible_patterns': [
                    r'feel.*faint', r'faint.*feel', r'dizzy', r'lightheaded', r'blackout',
                    r'pass.*out', r'lost.*consciousness', r'unconscious'
                ],
                'severity_keywords': {
                    'severe': ['lost consciousness', 'passed out', 'fainted', 'blackout', 'unconscious'],
                    'moderate': ['nearly fainted', 'almost passed out', 'severe dizziness', 'very dizzy'],
                    'mild': ['lightheaded', 'dizzy spells', 'feeling faint', 'little dizzy']
                },
                'base_severity': 'high',
                'follow_ups': ['circumstances', 'warning_signs', 'recovery_time', 'frequency']
            },
            'fatigue': {
                'keywords': [
                    'fatigue', 'tired', 'exhausted', 'weakness', 'weak', 'lack of energy',
                    'no energy', 'low energy', 'worn out'
                ],
                'flexible_patterns': [
                    r'feel.*tired', r'tired.*feel', r'feel.*weak', r'weak.*feel',
                    r'no.*energy', r'low.*energy', r'lack.*energy', r'exhausted',
                    r'worn.*out', r'fatigue'
                ],
                'severity_keywords': {
                    'severe': ['exhausted', 'extremely tired', 'no energy', 'can barely function', 'completely drained'],
                    'moderate': ['very tired', 'weak', 'low energy', 'more tired than usual', 'quite tired'],
                    'mild': ['slightly tired', 'minor fatigue', 'less energy', 'little tired', 'somewhat tired']
                },
                'base_severity': 'medium',
                'follow_ups': ['onset', 'progression', 'exertion_tolerance', 'rest_improvement']
            },
            'edema': {
                'keywords': [
                    'swelling', 'swollen legs', 'swollen feet', 'swollen ankles', 'fluid retention',
                    'swollen', 'puffy', 'bloated'
                ],
                'flexible_patterns': [
                    r'swollen.*legs', r'swollen.*feet', r'swollen.*ankle', r'legs.*swollen',
                    r'feet.*swollen', r'ankle.*swollen', r'swelling.*legs', r'swelling.*feet'
                ],
                'severity_keywords': {
                    'severe': ['severe swelling', 'very swollen', 'significant edema', 'extremely swollen'],
                    'moderate': ['noticeable swelling', 'moderate swelling', 'quite swollen'],
                    'mild': ['slight swelling', 'minor swelling', 'barely noticeable', 'little swollen']
                },
                'base_severity': 'medium',
                'follow_ups': ['location', 'timing', 'progression', 'associated_symptoms']
            }
        }

        # Severity descriptors for natural conversation
        self.severity_responses = {
            'severe': {
                'concern_level': 'very concerning',
                'urgency': 'needs immediate attention',
                'tone': 'This sounds quite serious'
            },
            'moderate': {
                'concern_level': 'concerning',
                'urgency': 'should be evaluated soon',
                'tone': 'This is certainly worth investigating'
            },
            'mild': {
                'concern_level': 'noteworthy',
                'urgency': 'should be monitored',
                'tone': 'While this seems mild'
            }
        }

    def get_initial_greeting(self):
        """Get the initial greeting message - this should be called only once"""
        return (
            "Hello! I'm Dr. Assistant, your cardiovascular health consultant. "
            "Thank you for taking the time to seek a professional assessment of your heart health. "
            "I'm here to listen carefully to your concerns and provide you with thorough guidance.\n\n"
            "To start our consultation, I'd like to understand what brings you here today. "
            "Could you please tell me about any symptoms, concerns, or reasons that prompted you to seek this cardiovascular assessment? "
            "Please feel free to describe everything you've been experiencing - I'm here to help."
        )

    def reset(self):
        """Reset the chatbot state for a new consultation."""
        self.consultation_data = {}
        self.current_stage = 'initial_greeting'
        self.follow_up_questions = []
        self.symptom_details = {}

    def process_response(self, user_input):
        """Process user response using professional medical consultation approach."""
        try:
            if not user_input.strip():
                return "I understand you may need a moment to think. Please take your time and share what's on your mind. I'm here to listen.", None

            # Get the current stage handler
            handler = self.consultation_stages.get(self.current_stage)
            if not handler:
                return "I apologize for the confusion. Let me help you properly. Could you please tell me what brings you here today?", None

            # Process the response through the appropriate stage handler
            response, next_stage, completed_data = handler(user_input.strip())
            
            # Update stage if provided
            if next_stage:
                self.current_stage = next_stage
                
            return response, completed_data

        except Exception as e:
            print(f"Error in process_response: {str(e)}")
            return "I apologize for the technical difficulty. Let me start fresh - could you please tell me what symptoms or concerns brought you here today?", None

    def _handle_initial_greeting(self, user_input):
        """Handle the initial greeting and transition to chief complaint."""
        # Skip to chief complaint processing since user already provided input
        return self._handle_chief_complaint(user_input)

    def _handle_chief_complaint(self, user_input):
        """Analyze the chief complaint with enhanced empathy and severity assessment."""
        self.consultation_data['chief_complaint'] = user_input
        
        # Analyze for symptoms and determine severity
        detected_symptoms = self._analyze_symptom_patterns_with_severity(user_input)
        
        if not detected_symptoms:
            if any(word in user_input.lower() for word in ['checkup', 'screening', 'prevention', 'worry', 'concerned', 'healthy', 'fine', 'normal']):
                response = (
                    "I really appreciate you being proactive about your cardiovascular health. That's exactly the right approach! "
                    "Even when we feel generally well, it's wise to check in on our heart health.\n\n"
                    "Now, sometimes our bodies give us subtle signals that we might not immediately connect to our heart. "
                    "In the past few weeks or months, have you noticed any changes in:\n\n"
                    "• How you feel during your usual activities\n"
                    "• Your energy levels or sleep patterns\n"
                    "• Any chest sensations, even very mild ones\n"
                    "• Your breathing during exercise or at rest\n"
                    "• Any unusual awareness of your heartbeat\n\n"
                    "Please share anything you've noticed, even if it seems minor. Sometimes the small details help us paint the complete picture."
                )
                return response, 'symptom_analysis', None
            else:
                response = (
                    "Thank you for sharing that with me. I want to make sure I understand your situation completely. "
                    "It sounds like you have some health concerns, and I'm here to help address them.\n\n"
                    "Could you help me understand if you're experiencing any physical symptoms right now? "
                    "For example, any sensations in your chest, changes in your breathing, unusual tiredness, "
                    "or anything else that feels different from your normal state?\n\n"
                    "Please describe anything you've been feeling, and don't worry about medical terminology - "
                    "just tell me in your own words what you've been experiencing."
                )
                return response, 'symptom_analysis', None
        
        # If symptoms detected, proceed with detailed analysis
        self.symptom_details = detected_symptoms
        most_concerning = self._get_most_concerning_symptom(detected_symptoms)
        severity_info = detected_symptoms[most_concerning]
        
        # Check for emergency symptoms first
        emergency_check = self._check_emergency_symptoms(user_input, detected_symptoms)
        if emergency_check:
            return emergency_check, 'assessment_summary', None
        
        # Personalized response based on symptom and severity
        symptom_name = most_concerning.replace('_', ' ')
        severity = severity_info.get('detected_severity', 'moderate')
        severity_response = self.severity_responses[severity]
        detected_phrase = severity_info.get('detected_phrase', symptom_name)
        
        response = (
            f"Thank you for telling me about the {detected_phrase} you're experiencing. {severity_response['tone']}, and I want to understand it thoroughly. "
            f"Based on your description, this appears to be {severity} {symptom_name}, which {severity_response['urgency']}.\n\n"
            f"To provide you with the most accurate assessment, I need to understand the specific characteristics of this symptom. "
            f"This will help me determine the best course of action for you.\n\n"
            f"Can you tell me more about:\n"
            f"• The exact location and how it feels (sharp, dull, pressing, burning, etc.)\n"
            f"• When you first noticed it and how long episodes typically last\n"
            f"• What seems to bring it on or make it worse\n"
            f"• What activities or positions affect it\n\n"
            f"Take your time and give me as much detail as you can - this information is very important for your assessment."
        )
        
        return response, 'symptom_details', None

    def _handle_symptom_analysis(self, user_input):
        """Handle general symptom analysis when no specific symptoms were initially detected."""
        if any(word in user_input.lower() for word in ['no', 'none', 'nothing', 'not really', 'feeling fine', 'no symptoms']):
            response = (
                "That's wonderful to hear that you're not experiencing any obvious symptoms! "
                "Your proactive approach to heart health is commendable.\n\n"
                "Since you're here for preventive care, let me ask about some subtle changes that sometimes occur gradually. "
                "Have you noticed any changes in your exercise tolerance? For instance:\n\n"
                "• Do you find yourself getting more winded during activities you used to do easily?\n"
                "• Have you had to slow down or take breaks during activities like climbing stairs?\n"
                "• Any changes in how you feel after physical exertion?\n\n"
                "Also, how has your energy level been overall? Any unusual fatigue or changes in your sleep patterns?\n\n"
                "These details help me understand your overall cardiovascular fitness."
            )
            return response, 'risk_factors', None
        
        # Re-analyze with broader approach
        detected_symptoms = self._analyze_symptom_patterns_with_severity(user_input)
        if detected_symptoms:
            self.symptom_details = detected_symptoms
            most_concerning = self._get_most_concerning_symptom(detected_symptoms)
            severity_info = detected_symptoms[most_concerning]
            
            symptom_name = most_concerning.replace('_', ' ')
            severity = severity_info.get('detected_severity', 'moderate')
            severity_response = self.severity_responses[severity]
            detected_phrase = severity_info.get('detected_phrase', symptom_name)
            
            response = (
                f"I see you're experiencing {detected_phrase}. {severity_response['tone']}, and I want to explore this carefully with you. "
                f"Based on what you've described, this appears to be {severity} {symptom_name} and {severity_response['urgency']}.\n\n"
                f"Let's discuss the pattern and characteristics of this symptom:\n\n"
                f"• When did you first notice this symptom?\n"
                f"• Has it been getting worse, staying the same, or improving?\n"
                f"• What activities or situations seem to trigger it?\n"
                f"• Have you found anything that helps relieve it?\n\n"
                f"Understanding these patterns will help me provide you with the most appropriate guidance."
            )
            return response, 'symptom_details', None
        else:
            response = (
                "Thank you for sharing that information with me. While you may not be experiencing classic cardiac symptoms, "
                "it's still important for me to understand your overall health picture.\n\n"
                "Let's discuss some important factors that can affect your heart health:\n\n"
                "Could you tell me about:\n"
                "• Any family history of heart disease, especially in parents or siblings\n"
                "• Whether you have conditions like high blood pressure, diabetes, or high cholesterol\n"
                "• Your current medications, if any\n"
                "• Your smoking history and current lifestyle habits\n\n"
                "This information helps me assess your cardiovascular risk profile comprehensively."
            )
            return response, 'risk_factors', None

    def _handle_symptom_details(self, user_input):
        """Handle detailed symptom characterization with medical expertise."""
        # Store the detailed symptom information
        if 'symptom_details_responses' not in self.consultation_data:
            self.consultation_data['symptom_details_responses'] = []
        self.consultation_data['symptom_details_responses'].append(user_input)
        
        # Determine if we need more symptom details or can move to medical history
        if len(self.consultation_data['symptom_details_responses']) < 2:
            if any(symptom in self.symptom_details for symptom in ['chest_pain', 'dyspnea', 'syncope']):
                response = (
                    "Thank you for those detailed descriptions - that's very helpful information. "
                    "Now I'd like to understand the timing and triggers better, as this can be quite revealing.\n\n"
                    "Could you help me understand:\n\n"
                    "• How long do these episodes typically last? (Seconds, minutes, hours?)\n"
                    "• What seems to bring them on? (Physical activity, stress, rest, certain positions, eating?)\n"
                    "• Does anything make them better? (Rest, medication, changing position?)\n"
                    "• Have you noticed any other symptoms that occur at the same time? (Nausea, sweating, dizziness?)\n\n"
                    "These patterns often give us important clues about what might be causing your symptoms."
                )
                return response, 'symptom_details', None
        
        # Move to medical history after gathering sufficient symptom details
        response = (
            "Excellent, thank you for providing such thorough information about your symptoms. "
            "This detailed history is invaluable for understanding your condition.\n\n"
            "Now, to complete my assessment, I need to understand your medical background. "
            "Could you please tell me about:\n\n"
            "• Any existing medical conditions you have (especially high blood pressure, diabetes, high cholesterol)\n"
            "• Previous heart problems or procedures you've had\n"
            "• Current medications you're taking (including vitamins and supplements)\n"
            "• Any allergies to medications\n"
            "• Other chronic conditions that affect your health\n\n"
            "This medical history helps me put your current symptoms in proper context."
        )
        return response, 'medical_history', None

    def _handle_medical_history(self, user_input):
        """Handle medical history collection with professional thoroughness."""
        self.consultation_data['medical_history'] = user_input
        
        response = (
            "Thank you for sharing your medical history. This background information is crucial for understanding your cardiovascular risk.\n\n"
            "Now I'd like to discuss some lifestyle and family factors that significantly impact heart health:\n\n"
            "• **Family History**: Any relatives with heart disease, heart attacks, or sudden cardiac death? (Especially parents, siblings, or children)\n"
            "• **Smoking**: Do you currently smoke or have you ever smoked? If former smoker, when did you quit?\n"
            "• **Physical Activity**: How would you describe your typical activity level? Any regular exercise?\n"
            "• **Stress**: Are you experiencing significant stress in your life currently?\n"
            "• **Diet**: How would you describe your typical eating habits?\n\n"
            "These factors help me understand your overall cardiovascular risk profile and provide targeted recommendations."
        )
        return response, 'risk_factors', None

    def _handle_risk_factors(self, user_input):
        """Handle risk factor assessment with comprehensive evaluation."""
        self.consultation_data['risk_factors'] = user_input
        
        response = (
            "Thank you for providing such comprehensive information. I now have a thorough understanding of your health status, "
            "symptoms, medical history, and risk factors.\n\n"
            "Let me take a moment to review everything you've shared and provide you with my professional assessment and recommendations. "
            "I'll give you a clear picture of what I think is happening and what steps we should take next."
        )
        return response, 'assessment_summary', None

    def _handle_assessment_summary(self, user_input):
        """Provide professional assessment summary with severity-based recommendations."""
        symptoms_present = bool(self.symptom_details)
        has_severe_symptoms = any(details.get('detected_severity') == 'severe' for details in self.symptom_details.values()) if symptoms_present else False
        has_moderate_symptoms = any(details.get('detected_severity') == 'moderate' for details in self.symptom_details.values()) if symptoms_present else False
        
        if has_severe_symptoms:
            response = (
                "**CLINICAL ASSESSMENT & RECOMMENDATIONS**\n\n"
                "Based on our comprehensive consultation, you have presented with **severe cardiovascular symptoms** that require urgent medical attention. "
                "The combination of symptoms you've described could indicate a serious cardiac condition that needs immediate evaluation.\n\n"
                "**⚠️ IMMEDIATE ACTION REQUIRED:**\n"
                "• **Seek emergency medical care within the next few hours**\n"
                "• Consider going to an emergency department if symptoms worsen\n"
                "• **Do not delay** - severe cardiac symptoms can be time-sensitive\n"
                "• Avoid any strenuous physical activity until medically cleared\n"
                "• Have someone available to assist you or drive you to medical care\n\n"
                "**RECOMMENDED MEDICAL EVALUATION:**\n"
                "• Comprehensive cardiovascular examination\n"
                "• ECG (electrocardiogram) and possibly stress testing\n"
                "• Blood work including cardiac enzyme levels\n"
                "• Echocardiogram or other cardiac imaging as indicated\n"
                "• Consultation with a cardiologist\n\n"
                "**Please remember**: This assessment provides guidance, but immediate professional medical evaluation is essential for proper diagnosis and treatment."
            )
        elif has_moderate_symptoms or symptoms_present:
            response = (
                "**CLINICAL ASSESSMENT & RECOMMENDATIONS**\n\n"
                "Based on our consultation, you have cardiovascular symptoms that, while not immediately life-threatening, "
                "warrant prompt medical evaluation to determine their cause and ensure appropriate management.\n\n"
                "**RECOMMENDED ACTIONS:**\n"
                "• **Schedule an appointment with your healthcare provider within 1-2 weeks**\n"
                "• Keep a detailed symptom diary noting triggers, duration, and severity\n"
                "• Continue normal daily activities but **avoid intense physical exertion**\n"
                "• Monitor symptoms closely and seek immediate care if they worsen\n"
                "• Consider stress management techniques\n\n"
                "**SUGGESTED MEDICAL WORKUP:**\n"
                "• Complete physical examination with focus on cardiovascular system\n"
                "• Baseline ECG and blood pressure monitoring\n"
                "• Basic metabolic panel and lipid profile\n"
                "• Possibly stress testing or echocardiogram based on clinical findings\n"
                "• Discussion of lifestyle modifications\n\n"
                "While your symptoms may not indicate an emergency, proper medical evaluation is important for peace of mind and appropriate treatment."
            )
        else:
            response = (
                "**CLINICAL ASSESSMENT & RECOMMENDATIONS**\n\n"
                "Based on our comprehensive consultation, you are not currently experiencing concerning cardiovascular symptoms. "
                "This is reassuring! Your proactive approach to cardiovascular health is exactly what we want to see.\n\n"
                "**PREVENTIVE RECOMMENDATIONS:**\n"
                "• **Continue regular cardiovascular health monitoring**\n"
                "• Schedule routine check-ups with your healthcare provider\n"
                "• Maintain heart-healthy lifestyle habits\n"
                "• Monitor blood pressure regularly if you have access\n"
                "• Stay alert to any new or changing symptoms\n\n"
                "**HEART-HEALTHY LIFESTYLE OPTIMIZATION:**\n"
                "• **Exercise**: Aim for 150 minutes of moderate aerobic activity weekly\n"
                "• **Nutrition**: Follow a heart-healthy diet (Mediterranean or DASH diet)\n"
                "• **Stress Management**: Practice stress reduction techniques\n"
                "• **Sleep**: Maintain good sleep hygiene (7-9 hours nightly)\n"
                "• **Avoid smoking** and limit alcohol consumption\n\n"
                "Continue monitoring your health and don't hesitate to seek medical attention if new symptoms develop."
            )
        
        return response, 'recommendations', self.consultation_data

    def _handle_recommendations(self, user_input):
        """Handle final recommendations and close consultation."""
        response = (
            "Thank you for taking the time to complete this comprehensive cardiovascular consultation with me. "
            "I hope our discussion has been helpful in understanding your heart health status and the appropriate next steps.\n\n"
            "**KEY TAKEAWAYS:**\n"
            "• You now have a clear assessment of your cardiovascular symptoms and risk factors\n"
            "• You have specific recommendations tailored to your situation\n"
            "• You understand the timeline for seeking further medical care\n\n"
            "**IMPORTANT REMINDERS:**\n"
            "• This consultation is educational and does not replace in-person medical care\n"
            "• **Always seek immediate medical attention** for severe or worsening symptoms\n"
            "• Follow up with healthcare providers as recommended\n"
            "• Continue monitoring your health and maintaining heart-healthy habits\n\n"
            "**Final Note**: If you have any concerns or if your symptoms change, please don't hesitate to consult with a qualified healthcare professional immediately. "
            "Your heart health is precious - take care of it and take care of yourself.\n\n"
            "Wishing you the very best in your journey toward optimal cardiovascular health!"
        )
        return response, 'completed', self.consultation_data

    def _analyze_symptom_patterns_with_severity(self, text):
        """Enhanced symptom analysis with flexible pattern matching and severity detection."""
        detected = {}
        text_lower = text.lower()
        
        for symptom_type, symptom_info in self.symptom_patterns.items():
            detected_phrase = None
            
            # First check exact keyword matches
            for keyword in symptom_info['keywords']:
                if keyword in text_lower:
                    detected_phrase = keyword
                    break
            
            # If no exact match, check flexible patterns using regex
            if not detected_phrase and 'flexible_patterns' in symptom_info:
                for pattern in symptom_info['flexible_patterns']:
                    if re.search(pattern, text_lower):
                        # Extract the matched phrase for better response
                        match = re.search(pattern, text_lower)
                        detected_phrase = match.group() if match else pattern.replace(r'.*', ' ').replace(r'\b', '')
                        break
            
            if detected_phrase:
                # Determine severity
                detected_severity = 'moderate'  # default
                
                # Check for severity indicators
                for severity, severity_keywords in symptom_info['severity_keywords'].items():
                    for severity_keyword in severity_keywords:
                        if severity_keyword in text_lower:
                            detected_severity = severity
                            break
                    if detected_severity != 'moderate':
                        break
                
                detected[symptom_type] = {
                    'base_severity': symptom_info['base_severity'],
                    'detected_severity': detected_severity,
                    'follow_ups': symptom_info['follow_ups'],
                    'detected_phrase': detected_phrase.strip()
                }
                break  # Only detect one primary symptom per input
                
        return detected

    def _get_most_concerning_symptom(self, detected_symptoms):
        """Determine the most concerning symptom based on severity."""
        severity_order = {'severe': 3, 'moderate': 2, 'mild': 1}
        base_severity_order = {'high': 3, 'medium': 2, 'low': 1}
        
        most_concerning = None
        highest_score = 0
        
        for symptom, details in detected_symptoms.items():
            detected_severity_score = severity_order.get(details.get('detected_severity', 'moderate'), 2)
            base_severity_score = base_severity_order.get(details.get('base_severity', 'medium'), 2)
            total_score = detected_severity_score + base_severity_score
            
            if total_score > highest_score:
                highest_score = total_score
                most_concerning = symptom
                
        return most_concerning

    def _check_emergency_symptoms(self, text, detected_symptoms):
        """Enhanced emergency symptom detection with severity consideration."""
        emergency_phrases = [
            'severe chest pain', 'crushing chest pain', 'chest pain radiating to arm',
            'chest pain with sweating', 'chest pain with nausea', 'can\'t breathe',
            'unable to breathe', 'passed out', 'lost consciousness', 'heart stopped',
            'crushing pain', 'tearing pain', 'excruciating pain'
        ]
        
        text_lower = text.lower()
        
        # Check for emergency phrases
        for phrase in emergency_phrases:
            if phrase in text_lower:
                return self._generate_emergency_response(phrase)
        
        # Check for severe symptoms
        for symptom, details in detected_symptoms.items():
            if details.get('detected_severity') == 'severe' and details.get('base_severity') == 'high':
                return self._generate_emergency_response(f"severe {symptom.replace('_', ' ')}")
        
        return None

    def _generate_emergency_response(self, trigger_phrase):
        """Generate emergency response with clear medical guidance."""
        return (
            "**🚨 MEDICAL EMERGENCY DETECTED 🚨**\n\n"
            f"Based on your description of {trigger_phrase}, this could represent a serious medical emergency requiring immediate attention.\n\n"
            "**TAKE THESE ACTIONS RIGHT NOW:**\n\n"
            "1. **Call emergency services (911/1122) IMMEDIATELY**\n"
            "2. **If experiencing chest pain**: Chew one aspirin (unless allergic) while waiting for help\n"
            "3. **Sit or lie down** in a comfortable position with head slightly elevated\n"
            "4. **Loosen tight clothing** around neck and chest\n"
            "5. **Have someone stay with you** - do NOT be alone\n"
            "6. **Do NOT drive yourself** to the hospital\n"
            "7. **Stay as calm as possible** while waiting for emergency services\n\n"
            "**⚠️ THIS IS NOT THE TIME TO WAIT OR HESITATE ⚠️**\n\n"
            "Emergency medical services can provide life-saving treatment en route to the hospital. "
            "Time is critical in cardiovascular emergencies - every minute counts.\n\n"
            "**Please call emergency services now and follow their instructions.**"
        )

# Maintain backward compatibility
class HeartDiseaseChatbot(ProfessionalHeartDoctorChatbot):
    """Backward compatibility wrapper for the enhanced professional chatbot."""
    pass 