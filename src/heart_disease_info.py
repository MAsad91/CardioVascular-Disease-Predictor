import random

class HeartDiseaseInfo:
    def __init__(self):
        self.risk_factors = {
            'age': {
                'description': 'Age is a significant risk factor for heart disease. The risk increases as you get older.',
                'tips': [
                    'Get regular check-ups as you age',
                    'Maintain a healthy lifestyle regardless of age',
                    'Be more vigilant about heart health after 45 for men and 55 for women'
                ]
            },
            'sex': {
                'description': 'Men are generally at higher risk of heart disease at younger ages, while women\'s risk increases after menopause.',
                'tips': [
                    'Women should be especially careful about heart health after menopause',
                    'Men should start regular heart health check-ups earlier',
                    'Both genders should maintain healthy lifestyle habits'
                ]
            },
            'cp': {
                'description': 'Chest pain (angina) can indicate heart problems. Different types of chest pain have different implications.',
                'tips': [
                    'Seek immediate medical attention for any chest pain',
                    'Learn to recognize different types of chest pain',
                    'Keep a record of chest pain episodes and triggers'
                ]
            },
            'trestbps': {
                'description': 'High blood pressure puts extra strain on your heart and blood vessels.',
                'tips': [
                    'Monitor your blood pressure regularly',
                    'Reduce salt intake',
                    'Maintain a healthy weight',
                    'Exercise regularly',
                    'Limit alcohol consumption'
                ]
            },
            'chol': {
                'description': 'High cholesterol can lead to plaque buildup in arteries, increasing heart disease risk.',
                'tips': [
                    'Eat a heart-healthy diet low in saturated fats',
                    'Exercise regularly',
                    'Maintain a healthy weight',
                    'Consider medication if lifestyle changes aren\'t enough'
                ]
            },
            'fbs': {
                'description': 'High fasting blood sugar can indicate diabetes, which increases heart disease risk.',
                'tips': [
                    'Monitor blood sugar levels regularly',
                    'Maintain a healthy diet',
                    'Exercise regularly',
                    'Keep weight in check',
                    'Follow medical advice for diabetes management'
                ]
            },
            'restecg': {
                'description': 'Electrocardiogram results can show heart rhythm and electrical activity abnormalities.',
                'tips': [
                    'Get regular ECG check-ups if recommended',
                    'Report any heart palpitations or irregular heartbeats',
                    'Follow up with your doctor about any abnormal results'
                ]
            },
            'thalach': {
                'description': 'Maximum heart rate during exercise can indicate heart health and fitness level.',
                'tips': [
                    'Exercise regularly to improve heart rate response',
                    'Monitor your heart rate during exercise',
                    'Stay within safe heart rate zones during workouts'
                ]
            },
            'exang': {
                'description': 'Exercise-induced chest pain can indicate reduced blood flow to the heart.',
                'tips': [
                    'Start exercise gradually and build up intensity',
                    'Stop exercising if you experience chest pain',
                    'Consult your doctor before starting a new exercise program'
                ]
            },
            'oldpeak': {
                'description': 'ST depression during exercise can indicate heart muscle stress or reduced blood flow.',
                'tips': [
                    'Follow your doctor\'s exercise recommendations',
                    'Monitor for any exercise-related symptoms',
                    'Get regular stress tests if recommended'
                ]
            },
            'slope': {
                'description': 'The slope of the ST segment during exercise can indicate heart health.',
                'tips': [
                    'Follow up with your doctor about any abnormal ECG results',
                    'Maintain regular exercise habits',
                    'Monitor for any exercise-related symptoms'
                ]
            },
            'ca': {
                'description': 'The number of major vessels affected by blockages indicates the severity of coronary artery disease.',
                'tips': [
                    'Follow your doctor\'s treatment plan',
                    'Make necessary lifestyle changes',
                    'Take prescribed medications regularly',
                    'Attend all follow-up appointments'
                ]
            },
            'thal': {
                'description': 'Thalassemia can affect blood flow and oxygen delivery to the heart.',
                'tips': [
                    'Follow your doctor\'s treatment plan',
                    'Maintain regular check-ups',
                    'Monitor for any new symptoms',
                    'Take prescribed medications regularly'
                ]
            }
        }

        self.general_tips = [
            'Exercise for at least 30 minutes most days of the week',
            'Eat a diet rich in fruits, vegetables, whole grains, and lean proteins',
            'Limit saturated and trans fats, cholesterol, and sodium',
            'Maintain a healthy weight',
            'Don\'t smoke and avoid secondhand smoke',
            'Limit alcohol consumption',
            'Manage stress through relaxation techniques',
            'Get 7-9 hours of sleep each night',
            'Monitor your blood pressure and cholesterol regularly',
            'Take prescribed medications as directed',
            'Stay hydrated by drinking plenty of water',
            'Limit processed foods and added sugars',
            'Practice portion control',
            'Include healthy fats like olive oil and avocados in your diet',
            'Get regular check-ups with your healthcare provider'
        ]

        self.warning_signs = [
            'Chest pain or discomfort (angina)',
            'Shortness of breath',
            'Pain in the neck, jaw, throat, upper abdomen, or back',
            'Pain, numbness, weakness, or coldness in legs or arms',
            'Irregular heartbeat',
            'Dizziness or lightheadedness',
            'Fatigue',
            'Swelling in legs, ankles, or feet',
            'Nausea or vomiting',
            'Sweating',
            'Indigestion or heartburn-like symptoms'
        ]

    def get_risk_factor_info(self, factor):
        return self.risk_factors.get(factor)

    def get_health_tips(self, risk_factors=None):
        tips = []
        if risk_factors:
            for factor in risk_factors:
                if factor in self.risk_factors:
                    tips.extend(self.risk_factors[factor]['tips'])
        tips.extend(self.general_tips)
        return list(set(tips))  # Remove duplicates

    def get_warning_signs(self):
        return self.warning_signs 