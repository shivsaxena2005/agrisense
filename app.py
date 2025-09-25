from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import requests
from dotenv import load_dotenv
from PIL import Image
from werkzeug.utils import secure_filename
import uuid
import base64
import random
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
load_dotenv()
# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','jfif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# DeepLeaf API Configuration
API_KEY = "feed_the_world_Q8GVdHWFV-mZzzWM1_jNDqU_Us-ZVL93EufRyAXQeso"
DEEPLEAF_URL = "https://api.deepleaf.io/analyze"

# Kindwise API Configuration for Crop Disease Detection
KINDWISE_API_KEY = "F2ffXDannBDAtNMEI4iDYluLT1lVLQF5D3B8VzftGlOUgDkW3u"
KINDWISE_API_BASE = "https://crop.kindwise.com/api/v1"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# 🖼 Validate + Resize Image
# ========================
def validate_and_resize(image_path):
    img = Image.open(image_path)
    width, height = img.size
    print("Original size:", width, "x", height)

    # If too small, resize to minimum 256x256
    if width < 200 or height < 200:
        img = img.resize((256, 256))
        new_path = "resized_" + os.path.basename(image_path)
        new_path = os.path.join(os.path.dirname(image_path), new_path)
        img.save(new_path)
        print("Image resized to:", img.size)
        return new_path
    # If too big, resize to max 2000x2000
    elif width > 2000 or height > 2000:
        img.thumbnail((2000, 2000))
        new_path = "resized_" + os.path.basename(image_path)
        new_path = os.path.join(os.path.dirname(image_path), new_path)
        img.save(new_path)
        print("Image resized to:", img.size)
        return new_path
    else:
        return image_path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========================
# 🔬 Kindwise API Disease Detection
# ========================
def analyze_with_kindwise(image_path, plant_id=None, latitude=None, longitude=None):
    """Analyze crop image using Kindwise API for disease detection"""
    try:
        # Convert image to base64
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare request data
        request_data = {
            "images": [base64_image]
        }
        
        # Add optional parameters
        if plant_id:
            request_data["plant_id"] = plant_id
        if latitude:
            request_data["latitude"] = latitude
        if longitude:
            request_data["longitude"] = longitude
        
        # Make request to Kindwise API
        headers = {
            'Api-Key': KINDWISE_API_KEY,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{KINDWISE_API_BASE}/identification',
            json=request_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            api_result = response.json()
            print(f"✅ Kindwise API Success! Status: {response.status_code}")
            print(f"📊 API Response: {api_result.get('result', {}).get('disease', {}).get('suggestions', [])[:2]}")
            return parse_kindwise_response(api_result)
        else:
            print(f"❌ Kindwise API Error: {response.status_code} - {response.text}")
            raise Exception(f"Kindwise API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error calling Kindwise API: {e}")
        return None

def parse_kindwise_response(api_result):
    """Parse Kindwise API response into our format"""
    try:
        result = api_result.get('result', {})
        diseases = result.get('disease', {}).get('suggestions', [])
        crops = result.get('crop', {}).get('suggestions', [])
        
        print(f"🔍 Parsing API Response:")
        print(f"   Diseases found: {len(diseases)}")
        print(f"   Crops found: {len(crops)}")
        if diseases:
            print(f"   Top disease: {diseases[0].get('name')} ({diseases[0].get('probability', 0)*100:.1f}%)")
        if crops:
            print(f"   Top crop: {crops[0].get('name')} ({crops[0].get('probability', 0)*100:.1f}%)")
        
        if not diseases:
            return {
                'disease': 'No diseases detected',
                'confidence': 'N/A',
                'severity': 'Healthy',
                'treatment': 'No treatment needed',
                'prevention': 'Continue good agricultural practices',
                'plant_name': crops[0].get('name', 'Unknown Plant') if crops else 'Unknown Plant',
                'scientific_name': crops[0].get('scientific_name', 'Unknown') if crops else 'Unknown'
            }
        
        # Get the most confident disease prediction
        main_disease = diseases[0]
        crop_info = crops[0] if crops else {}
        
        # Generate treatment and prevention suggestions
        disease_name = main_disease.get('name', 'Unknown Disease')
        scientific_name = main_disease.get('scientific_name', 'Unknown')
        confidence = main_disease.get('probability', 0) * 100
        
        treatment_suggestions = generate_treatment_suggestions(disease_name)
        prevention_suggestions = generate_prevention_suggestions(disease_name)
        
        return {
            'disease': disease_name,
            'scientific_name': scientific_name,
            'confidence': f"{confidence:.1f}%",
            'severity': 'High' if confidence > 70 else 'Medium' if confidence > 40 else 'Low',
            'treatment': treatment_suggestions,
            'prevention': prevention_suggestions,
            'plant_name': crop_info.get('name', 'Unknown Plant'),
            'plant_scientific_name': crop_info.get('scientific_name', 'Unknown'),
            'all_diseases': diseases[:3]  # Top 3 disease predictions
        }
        
    except Exception as e:
        print(f"Error parsing Kindwise response: {e}")
        return None

def generate_treatment_suggestions(disease_name):
    """Generate treatment suggestions based on disease name"""
    treatments = {
        'early blight': 'Apply copper-based fungicide, remove infected leaves, improve air circulation',
        'late blight': 'Apply systemic fungicide, remove infected plants, avoid overhead watering',
        'powdery mildew': 'Apply sulfur-based fungicide, improve air circulation, reduce humidity',
        'downy mildew': 'Apply copper fungicide, improve drainage, remove infected debris',
        'anthracnose': 'Apply fungicide, remove infected plant parts, improve sanitation',
        'bacterial leaf spot': 'Apply copper spray, avoid overhead watering, remove infected plants',
        'rust': 'Apply fungicide, remove infected leaves, use resistant varieties',
        'mosaic virus': 'Remove infected plants, control aphids, use virus-free seeds'
    }
    
    disease_lower = disease_name.lower()
    for key, treatment in treatments.items():
        if key in disease_lower:
            return treatment
    
    return 'Consult with agricultural extension service for specific treatment recommendations'

def generate_prevention_suggestions(disease_name):
    """Generate prevention suggestions based on disease name"""
    preventions = {
        'early blight': 'Crop rotation, proper spacing, avoid overhead watering, remove plant debris',
        'late blight': 'Plant resistant varieties, avoid overhead watering, proper spacing',
        'powdery mildew': 'Improve air circulation, avoid overcrowding, morning watering',
        'downy mildew': 'Improve drainage, avoid overhead watering, proper spacing',
        'anthracnose': 'Crop rotation, remove plant debris, avoid overhead watering',
        'bacterial leaf spot': 'Avoid overhead watering, proper spacing, crop rotation',
        'rust': 'Plant resistant varieties, remove infected debris, proper spacing',
        'mosaic virus': 'Use virus-free seeds, control aphids, remove weeds'
    }
    
    disease_lower = disease_name.lower()
    for key, prevention in preventions.items():
        if key in disease_lower:
            return prevention
    
    return 'Maintain good agricultural practices, proper spacing, and regular monitoring'

def run_langchain_pipeline(disease_name: str, language: int = 0) -> str:
    """
    Run LangChain farming advice pipeline.
    Args:
        disease_name (str): The disease name detected from image.
        language (int): 0 = English, 1 = Hindi.
    Returns:
        str: Final output in the chosen language.
    """
    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            return "GROQ API key not found. Please check your environment variables."
        
        # LLMs
        llm_groq_creative = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=GROQ_API_KEY)
        llm_groq_non_creative = ChatGroq(model="llama-3.1-8b-instant", temperature=0.05, api_key=GROQ_API_KEY)

        # Prompt templates
        farming_expert_prompt = PromptTemplate.from_template(
            """You are an expert in agriculture and farming practices in India. 
            Answer the user's farming-related question with detailed, practical advice for India.
            
            Disease: {disease_name}
            
            Provide a detailed farming-related answer (around 100 words) structured under:
            - About Disease
            - Precaution 
            - Prevention
            
            Answer:"""
        )

        content_shortener_prompt = PromptTemplate.from_template(
            """You are skilled at condensing long farming explanations into short, farmer-friendly tips.
            
            Take the following detailed farming advice and shorten it into a simple, clear response 
            in 2–3 sentences, formatted under 'About Disease', 'Precaution' and 'Prevention' with step-wise guidance 
            so a farmer can easily understand and follow.
            
            Detailed advice: {farming_advice}
            
            Format your response as:
            About Disease: ...
            Precaution:
            1. ...
            2. ...
            Prevention:
            1. ...
            2. ...
            
            Shortened advice:"""
        )

        translator_prompt = PromptTemplate.from_template(
            """You are an expert in translating English farming instructions into clear Hindi for farmers.
            
            Translate the following summarized farming advice into Hindi without changing the context or meaning:
            
            {content}
            
            Format your response as:
            About Disease: ... (in Hindi)
            Precaution:
            1. ... (in Hindi)
            2. ... (in Hindi)
            Prevention:
            1. ... (in Hindi)
            2. ... (in Hindi)
            
            Hindi translation:"""
        )

        # Create chains using LCEL (LangChain Expression Language)
        output_parser = StrOutputParser()
        
        # Step 1: Get farming advice
        farming_chain = farming_expert_prompt | llm_groq_non_creative | output_parser
        farming_advice = farming_chain.invoke({"disease_name": disease_name})
        
        # Step 2: Shorten the advice
        shortener_chain = content_shortener_prompt | llm_groq_non_creative | output_parser
        shortened_advice = shortener_chain.invoke({"farming_advice": farming_advice})

        # If Hindi translation is required
        if language == 1:
            translator_chain = translator_prompt | llm_groq_non_creative | output_parser
            hindi_result = translator_chain.invoke({"content": shortened_advice})
            return hindi_result
        else:
            return shortened_advice
            
    except Exception as e:
        print(f"LangChain pipeline error: {e}")
        return f"Error processing disease information: {str(e)}"

# Mock data generators
def get_iot_data():
    return {
        'soil_health': {
            'ph': round(random.choice([6.0, 7.5, 7.8, 5.5, 5.8, 6.4, 7.0])),
            'nutrients': random.choice(['Optimal', 'Good', 'Fair']),
            'moisture': random.choice([60, 85, 45, 70, 90, 50])
        },
        'plant_health': {
            'growth': random.choice(['Healthy', 'Good', 'Moderate']),
            'stress': random.choice(['None', 'Low', 'Moderate'])
        },
        'weather_alert': {
            'type': random.choice(['Weahter Alert']),
            'warning': random.choice([
                'The weather is sunny'
            ])
        }
    }

def get_growth_predictions():
    return {
        'planting': {
            'window': 'March 15 - April 15',
            'optimal_date': 'March 28'
        },
        'harvesting': {
            'window': 'July 15 - August 15',
            'estimated_date': 'July 28'
        },
        'yield': {
            'expected': random.randint(1000, 1500),
            'progress': random.randint(70, 95)
        }
    }

def get_crop_varieties():
    return [
        {
            'name': 'Drought-Resistant Maize',
            'description': 'High yield in arid conditions',
            'icon': '🌽',
            'color': 'teal'
        },
        {
            'name': 'Hybrid Tomato',
            'description': 'Resistant to common diseases',
            'icon': '🍅',
            'color': 'green'
        },
        {
            'name': 'Organic Wheat',
            'description': 'Grows well in nutrient-rich soil',
            'icon': '🌾',
            'color': 'brown'
        }
    ]

def get_market_data():
    commodities = [
        {'name': 'Wheat', 'market': 'Delhi', 'price': random.randint(2000, 2500), 'arrivals': random.randint(4000, 6000), 'time': '10:30 AM'},
        {'name': 'Rice', 'market': 'Mumbai', 'price': random.randint(1800, 2200), 'arrivals': random.randint(3500, 5000), 'time': '11:00 AM'},
        {'name': 'Cotton', 'market': 'Ahmedabad', 'price': random.randint(6000, 7000), 'arrivals': random.randint(2500, 3500), 'time': '10:45 AM'},
        {'name': 'Soybean', 'market': 'Indore', 'price': random.randint(4000, 4500), 'arrivals': random.randint(2000, 3000), 'time': '11:15 AM'},
        {'name': 'Sugarcane', 'market': 'Lucknow', 'price': random.randint(250, 350), 'arrivals': random.randint(8000, 12000), 'time': '10:00 AM'},
    ]
    return commodities


# Routes
# Language content dictionaries
LANGUAGE_CONTENT = {
    'en': {
        'nav_dashboard': 'Dashboard',
        'nav_disease': 'Disease',
        'nav_iot': 'IoT & Growth',
        'nav_chat': 'Chat Model',
        'nav_market': 'Market & Partnerships',
        'upload_btn': '📤 Upload Crop Image',
        'disease_diagnosis': 'Disease Diagnosis',
        'iot_monitoring': 'IoT Monitoring',
        'growth_predictions': 'Growth Predictions',
        'market_data': 'Market Data',
        'ai_analysis': '🔬 AI Disease Analysis & Recommendations',
        'soil_health': 'Soil Health',
        'plant_health': 'Plant Health',
        'weather_alert': 'Weather Alert',
        'ph_level': 'pH Level',
        'nutrients': 'Nutrients',
        'moisture': 'Moisture',
        'growth_status': 'Growth Status',
        'stress_level': 'Stress Level',
        'alert_type': 'Alert Type',
        'planting_window': 'Planting Window',
        'harvest_time': 'Harvest Time',
        'expected_yield': 'Expected Yield',
        'kg_acre': 'kg/acre',
        'upload_placeholder': 'Upload image for AI diagnosis',
        'no_analysis': 'No Analysis Available',
        'upload_instruction': 'Upload a crop image to get detailed disease analysis and recommendations.',
        'go_back_dashboard': 'Go Back to Dashboard',
        'market_dashboard': 'Market Dashboard',
        'iot_monitoring': 'IoT Monitoring & Growth Prediction',
        'commodity_prices': 'Commodity Prices & Arrivals',
        'wheat_price_trend': 'Wheat Price Trend',
        'total_daily_arrivals': 'Total Daily Arrivals',
        'top_performer': 'Top Performer',
        'commodity': 'Commodity',
        'market': 'Market',
        'price_per_quintal': 'Price (per quintal)',
        'arrivals_bags': 'Arrivals (bags)',
        'last_updated': 'Last Updated',
        'soil_health': 'Soil Health',
        'plant_health': 'Plant Health',
        'weather_alert': 'Weather Alert',
        'growth_predictions': 'Growth Predictions',
        'crop_varieties': 'Crop Varieties',
        'recommended_varieties': 'Recommended Varieties',
        'drought_resistant': 'Drought-Resistant',
        'high_yield': 'High Yield',
        'disease_resistant': 'Disease-Resistant',
        'organic': 'Organic',
        'disease_detection_title': 'Plant Disease Detection & Analysis',
        'disease_detection_description': 'Upload an image of your crop to get AI-powered disease detection, prevention tips, and treatment recommendations.',
        'upload_crop_image': 'Upload Crop Image',
        'disease_analysis_results': 'Disease Analysis Results',
        'plant_identified': 'Plant Identified',
        'disease_detected': 'Disease Detected',
        'treatment_recommendations': 'Treatment Recommendations',
        'prevention_measures': 'Prevention Measures',
        'disease': 'Disease',
        'host_example': 'Host (example)',
        'symptoms': 'Symptoms',
        'quick_solution': 'Quick solution (in a few words)',
        'current_field_readings': 'Current Field Readings',
        'soil_moisture': 'Soil Moisture',
        'soil_ph': 'Soil pH',
        'nutrient_level': 'Nutrient Level',
        'plant_growth': 'Plant Growth',
        'stress_level': 'Stress Level',
        'growth_timeline_predictions': 'Growth Timeline & Predictions'
    },
    'hi': {
        'nav_dashboard': 'डैशबोर्ड',
        'nav_disease': 'रोग',
        'nav_chat' : 'Model',
        'nav_iot': 'IoT और विकास',
        'nav_market': 'बाजार और साझेदारी',
        'upload_btn': '📤 फसल छवि अपलोड करें',
        'disease_diagnosis': 'रोग निदान',
        'iot_monitoring': 'IoT निगरानी',
        'growth_predictions': 'विकास भविष्यवाणी',
        'market_data': 'बाजार डेटा',
        'ai_analysis': '🔬 AI रोग विश्लेषण और सिफारिशें',
        'soil_health': 'मिट्टी का स्वास्थ्य',
        'plant_health': 'पौधे का स्वास्थ्य',
        'weather_alert': 'मौसम चेतावनी',
        'ph_level': 'pH स्तर',
        'nutrients': 'पोषक तत्व',
        'moisture': 'नमी',
        'growth_status': 'विकास स्थिति',
        'stress_level': 'तनाव स्तर',
        'alert_type': 'चेतावनी प्रकार',
        'planting_window': 'रोपण विंडो',
        'harvest_time': 'कटाई का समय',
        'expected_yield': 'अपेक्षित उपज',
        'kg_acre': 'किलो/एकड़',
        'upload_placeholder': 'AI निदान के लिए छवि अपलोड करें',
        'no_analysis': 'कोई विश्लेषण उपलब्ध नहीं',
        'upload_instruction': 'विस्तृत रोग विश्लेषण और सिफारिशें प्राप्त करने के लिए फसल की छवि अपलोड करें।',
        'go_back_dashboard': 'डैशबोर्ड पर वापस जाएं',
        'market_dashboard': 'बाजार डैशबोर्ड',
        'iot_monitoring': 'IoT निगरानी और विकास भविष्यवाणी',
        'commodity_prices': 'कमोडिटी कीमतें और आगमन',
        'wheat_price_trend': 'गेहूं कीमत प्रवृत्ति',
        'total_daily_arrivals': 'कुल दैनिक आगमन',
        'top_performer': 'शीर्ष प्रदर्शक',
        'commodity': 'कमोडिटी',
        'market': 'बाजार',
        'price_per_quintal': 'कीमत (प्रति क्विंटल)',
        'arrivals_bags': 'आगमन (बैग)',
        'last_updated': 'अंतिम अपडेट',
        'soil_health': 'मिट्टी का स्वास्थ्य',
        'plant_health': 'पौधे का स्वास्थ्य',
        'weather_alert': 'मौसम चेतावनी',
        'growth_predictions': 'विकास भविष्यवाणी',
        'crop_varieties': 'फसल किस्में',
        'recommended_varieties': 'अनुशंसित किस्में',
        'drought_resistant': 'सूखा प्रतिरोधी',
        'high_yield': 'उच्च उपज',
        'disease_resistant': 'रोग प्रतिरोधी',
        'organic': 'जैविक',
        'disease_detection_title': 'पौधे की बीमारी का पता लगाना और विश्लेषण',
        'disease_detection_description': 'AI-संचालित बीमारी का पता लगाने, रोकथाम के सुझाव और उपचार की सिफारिशें प्राप्त करने के लिए अपनी फसल की एक छवि अपलोड करें।',
        'upload_crop_image': 'फसल की छवि अपलोड करें',
        'disease_analysis_results': 'बीमारी विश्लेषण परिणाम',
        'plant_identified': 'पौधा पहचाना गया',
        'disease_detected': 'बीमारी का पता चला',
        'treatment_recommendations': 'उपचार सिफारिशें',
        'prevention_measures': 'रोकथाम उपाय',
        'disease': 'बीमारी',
        'host_example': 'होस्ट (उदाहरण)',
        'symptoms': 'लक्षण',
        'quick_solution': 'त्वरित समाधान (कुछ शब्दों में)',
        'current_field_readings': 'वर्तमान फील्ड रीडिंग',
        'soil_moisture': 'मिट्टी की नमी',
        'soil_ph': 'मिट्टी का pH',
        'nutrient_level': 'पोषक तत्व स्तर',
        'plant_growth': 'पौधे की वृद्धि',
        'stress_level': 'तनाव स्तर',
        'growth_timeline_predictions': 'विकास समयरेखा और भविष्यवाणी'
    }
}

@app.route('/',methods=['GET', 'POST'])
def dashboard():
    langchain_result = None
    diagnosis_result = None
    uploaded_image = None
    
    # Get language from URL parameter (works for both GET and POST)
    language = request.args.get('lang', 'en')  # Default to English
    
    if(request.method == 'POST'):
        # Handle image upload
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename != '' and allowed_file(file.filename):
                # Generate unique filename and save file
                filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_image = filename
                
                # Process with Kindwise API for disease detection
                try:
                    # Get optional parameters from form
                    plant_id = request.form.get('plant_id')
                    latitude = request.form.get('latitude')
                    longitude = request.form.get('longitude')
                    
                    # Convert to appropriate types
                    if latitude:
                        latitude = float(latitude)
                    if longitude:
                        longitude = float(longitude)
                    
                    # Analyze with Kindwise API
                    diagnosis_result = analyze_with_kindwise(filepath, plant_id, latitude, longitude)
                    
                    if diagnosis_result:
                        # Get language preference from form (0 = English, 1 = Hindi)
                        language_code = int(request.form.get('language', 0))
                        # Update language based on form input
                        language = 'hi' if language_code == 1 else 'en'
                        
                        # Run LangChain pipeline with detected disease
                        disease_name = diagnosis_result.get('disease', 'Unknown Disease')
                        langchain_result = run_langchain_pipeline(disease_name, language_code)
                        
                        print(f"\n=== LangChain Output ({'Hindi' if language_code == 1 else 'English'}) ===")
                        print(langchain_result)
                    else:
                        raise Exception("Failed to get analysis results from Kindwise API")
                        
                except Exception as e:
                    print(f"Error processing image: {e}")
                    flash(f"Error analyzing image: {e}")
                    # Use mock data as fallback
                    diagnosis_result = {
                        'disease': 'Early Blight (Fallback)',
                        'scientific_name': 'Alternaria solani',
                        'severity': 'High',
                        'confidence': '92%',
                        'treatment': 'Apply copper-based fungicide, remove infected leaves, improve air circulation',
                        'prevention': 'Crop rotation, proper spacing, avoid overhead watering, remove plant debris',
                        'plant_name': 'Unknown Plant',
                        'plant_scientific_name': 'Unknown'
                    }
                    # Still run LangChain with fallback disease
                    language_code = int(request.form.get('language', 0))
                    # Update language based on form input
                    language = 'hi' if language_code == 1 else 'en'
                    langchain_result = run_langchain_pipeline(diagnosis_result['disease'], language_code)
            else:
                flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')

    iot_data = get_iot_data()
    predictions = get_growth_predictions()
    
    return render_template('dashboard.html', 
                           iot_data=iot_data,
                           predictions=predictions,
                           uploaded_image=uploaded_image,
                           diagnosis_result=diagnosis_result,
                           langchain_result=langchain_result,
                           language=language,
                           lang=LANGUAGE_CONTENT[language])


@app.route('/iot')
def iot_monitoring():
    language = request.args.get('lang', 'en')
    iot_data = get_iot_data()
    predictions = get_growth_predictions()
    crop_varieties = get_crop_varieties()
    
    return render_template('iot_monitoring_growthing.html', 
                           iot_data=iot_data,
                           predictions=predictions,
                           crop_varieties=crop_varieties,
                           language=language,
                           lang=LANGUAGE_CONTENT[language])

@app.route('/market')
def market_dashboard():
    language = request.args.get('lang', 'en')
    market_data = get_market_data()
    
    # Calculate price trends
    wheat_trend = {
        'change': '+2.5%',
        'direction': 'up',
        'last_30_days': '+2.5%'
    }
    
    return render_template('market_dashboard.html', 
                           market_data=market_data,
                           wheat_trend=wheat_trend,
                           language=language,
                           lang=LANGUAGE_CONTENT[language])

# Disease detection is now integrated into the main dashboard


def chat1(query):
        user_message = query
        language = request.args.get('lang', 'en')
        # Here you would typically process the message and generate a response

        try:
            
# -------------------- Agentic System Setup --------------------
# Load API key
            load_dotenv()
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq LLMs
            llm_groq_creative = ChatGroq(model="openai/gpt-oss-20b",
            api_key=GROQ_API_KEY,
            temperature=0.7
            )

            llm_groq_non_creative = ChatGroq(
                model="openai/gpt-oss-20b",
                api_key=GROQ_API_KEY,
                temperature=0
            )

            # Validation chain
            validator_prompt = PromptTemplate(
                input_variables=["user_input"],
                template=(
                    "Check if the following input is about farming/agriculture. "
                    "If yes, respond with only 'Relevant'. If no, respond with only 'Not relevant'.\n\n"
                    "Input: {user_input}"
                )
            )
            validator_chain = LLMChain(llm=llm_groq_non_creative, prompt=validator_prompt)

            # Farming expert chain
            farming_prompt = PromptTemplate(
                input_variables=["user_input", "validation_result"],
                template=(
                    "Validation result: {validation_result}\n\n"
                    "If validation_result is 'Relevant', answer the farming-related query for India:\n"
                    "{user_input}\n\n"
                    "If validation_result is 'Not relevant', reply with: "
                    "'This query is outside the farming domain.'"
                )
            )
            farming_chain = LLMChain(llm=llm_groq_non_creative, prompt=farming_prompt)

            # Summarizer chain
            summarizer_prompt = PromptTemplate(
                input_variables=["farming_answer"],
                template=(
                    "You are an Indian agriculture assistant. Summarize the following farming advice into a "
                    "single short paragraph (max 2 sentences, under 40 words). "
                    "Avoid extra details, tables, or long explanations.\n\n"
                    "Advice:\n{farming_answer}\n\n"
                    "Final short answer:"
                )
            )
            summarizer_chain = LLMChain(llm=llm_groq_non_creative, prompt=summarizer_prompt)

            # Formatter chain
            formatter_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                "You are a formatter who takes farming advice and makes it structured, "
                "clear, and farmer-friendly. Always organize the response into sections "
                "with headings and emojis when useful. Use short sentences, bullets, or "
                "tables for readability.\n\n"
                "Rules:\n"
                "1. Keep the output simple and easy to scan.\n"
                "2. Always group related points together under clear headings.\n"
                "3. If data is numeric (prices, rainfall, yield, etc.), prefer a table.\n"
                "4. If data is step-by-step, use numbered lists.\n"
                "5. Avoid long paragraphs — focus on clarity.\n"
                "6. **Maximum total length of your response must be 30 words**. "
                "Do not exceed this word limit."),
                ("human", "{summary}")
            ])
            formatter_chain = formatter_prompt | llm_groq_non_creative | StrOutputParser()

# Function to run the agentic system
            #def run_agentic_system(user_input: str, save_md: bool = True, md_path: str = "output.md"):
    # 1. Validation
            validation_result = validator_chain.invoke({"user_input": user_message})
    
            if "Not relevant" in validation_result:
                final_output = "This query is outside the farming domain."
            # if save_md:
            #     with open(md_path, "w", encoding="utf-8") as f:
            #         f.write(f"# Output\n\n{final_output}")
            # return final_output

    # 2. Farming Expert Answer
            farming_answer = farming_chain.invoke({
            "user_input": user_message,
            "validation_result": validation_result
            })

    # 3. Summarizer
            summary = summarizer_chain.invoke({"farming_answer": farming_answer})

    # 4. Structured Formatter
            structured_output = formatter_chain.invoke({"summary": summary})

    # Truncate to 30 words
            words = structured_output.split()
            final_output = " ".join(words[:30]) + ("..." if len(words) > 30 else "")
            #history.append({"User": user_message,"AI" :final_output})
            return(final_output)
    # Save to Markdown if needed
            # if save_md:
            #     with open(md_path, "w", encoding="utf-8") as f:
            #     f.write(f"# Structured Output (max 30 words)\n\n{final_output}")
        except Exception as e:
            print(f"Error in agentic system: {e}")
            final_output = "Error processing your request. Please try again later."
            redirect(url_for('chat',methods = ['POST'],user_message = user_message))
        return final_output
    
    #return render_template('chat.html', language=language, lang=LANGUAGE_CONTENT[language])

# -------------------- STT Function --------------------
# @app.route('/voice_input', methods=['POST'])
# def get_voice_input(duration=7):
#     import speech_recognition as sr
#     if(request.method == 'POST' ):
#         r = sr.Recognizer()
#         with sr.Microphone() as source:
#             print(f"🎙️ Speak something (max {duration} seconds)...")
#             try:
#                 audio = r.listen(source, timeout=duration, phrase_time_limit=duration)
#                 try:
#                     text = r.recognize_google(audio)
#                     print("📝 You said:", text)
#                     return (redirect(url_for('chat',methods = ['POST'],user_message = text)))
#                 except Exception as e:
#                     print("❌ Error recognizing speech:", e)
#                     return ""
#             except sr.WaitTimeoutError:
#                 print("⏱️ No speech detected within the time limit.")
#             return ""
#                 # except sr.RequestError as e:
#             print(f"❌ STT request failed: {e}")
#             return ""




# ... (All your existing app configuration and functions) ...

# -------------------- Agentic System Setup for Chatbot --------------------
# This is the corrected version of your chatbot logic, placed outside the route function
# # to prevent re-initializing it on every request.
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# llm_groq_creative = ChatGroq(
#     model="llama-3.1-8b-instant",  # Corrected model name
#     api_key=GROQ_API_KEY,
#     temperature=0.7
# )

# llm_groq_non_creative = ChatGroq(
#     model="llama-3.1-8b-instant",  # Corrected model name
#     api_key=GROQ_API_KEY,
#     temperature=0
# )

# # Validation chain
# validator_prompt = PromptTemplate(
#     input_variables=["user_input"],
#     template=(
#         "Check if the following input is about farming/agriculture. "
#         "If yes, respond with only 'Relevant'. If no, respond with only 'Not relevant'.\n\n"
#         "Input: {user_input}"
#     )
# )
# validator_chain = LLMChain(llm=llm_groq_non_creative, prompt=validator_prompt)

# # Farming expert chain
# farming_prompt = PromptTemplate(
#     input_variables=["user_input", "validation_result"],
#     template=(
#         "Validation result: {validation_result}\n\n"
#         "If validation_result is 'Relevant', answer the farming-related query for India:\n"
#         "{user_input}\n\n"
#         "If validation_result is 'Not relevant', reply with: "
#         "'This query is outside the farming domain.'"
#     )
# )
# farming_chain = LLMChain(llm=llm_groq_non_creative, prompt=farming_prompt)

# # Summarizer chain
# summarizer_prompt = PromptTemplate(
#     input_variables=["farming_answer"],
#     template=(
#         "You are an Indian agriculture assistant. Summarize the following farming advice into a "
#         "single short paragraph (max 2 sentences, under 40 words). "
#         "Avoid extra details, tables, or long explanations.\n\n"
#         "Advice:\n{farming_answer}\n\n"
#         "Final short answer:"
#     )
# )
# summarizer_chain = LLMChain(llm=llm_groq_non_creative, prompt=summarizer_prompt)

# # Formatter chain
# formatter_prompt = ChatPromptTemplate.from_messages([
#     ("system", 
#     "You are a formatter who takes farming advice and makes it structured, "
#     "clear, and farmer-friendly. Always organize the response into sections "
#     "with headings and emojis when useful. Use short sentences, bullets, or "
#     "tables for readability.\n\n"
#     "Rules:\n"
#     "1. Keep the output simple and easy to scan.\n"
#     "2. Always group related points together under clear headings.\n"
#     "3. If data is numeric (prices, rainfall, yield, etc.), prefer a table.\n"
#     "4. If data is step-by-step, use numbered lists.\n"
#     "5. Avoid long paragraphs — focus on clarity.\n"
#     "6. **Maximum total length of your response must be 30 words**. "
#     "Do not exceed this word limit."),
#     ("human", "{summary}")
# ])
# formatter_chain = formatter_prompt | llm_groq_non_creative | StrOutputParser()

# def run_agentic_system(user_input: str) -> str:
#     """Runs the chatbot logic based on your provided agentic system."""
#     try:
#         validation_result = validator_chain.invoke({"user_input": user_input})
#         if "Not relevant" in validation_result:
#             return "This query is outside the farming domain."
        
#         farming_answer = farming_chain.invoke({
#             "user_input": user_input,
#             "validation_result": validation_result
#         })
        
#         summary = summarizer_chain.invoke({"farming_answer": farming_answer})
#         structured_output = formatter_chain.invoke({"summary": summary})
        
#         words = structured_output.split()
#         final_output = " ".join(words[:30]) + ("..." if len(words) > 30 else "")
#         return final_output
#     except Exception as e:
#         print(f"Chatbot API error: {e}")
#         return 'Sorry, an error occurred while processing your request. Please try again later.'


# The history list should be stored in the session, not a global variable.
# history = [] # REMOVE this line
history = [{'User': '''''', 'AI': '''Hello!,I am a Farming Assistant'''}]
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Make sure 'chat_history' exists in the session
    language = request.args.get('lang', 'en')
    global history
    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            # Process the message and get the AI's response
            ai_response = chat1(user_message)
            
            # Append both messages to the history
            history.append({"User": user_message, "AI": ai_response})
            session.modified = True  # Tell Flask the session has changed
        
        # After processing, redirect back to the chat page to prevent re-submitting the form
        return redirect(url_for('chat'))

    # For a GET request, just render the chat page with the current history
    return render_template('chat.html', history=history, language=language, lang=LANGUAGE_CONTENT[language])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    history = []
    return redirect(url_for('chat'))

@app.route('/voice_input', methods=['POST'])
def get_voice_input():
    # This route will process a POST request from a voice form
    # The 'request.form.get('message')' will get the transcribed text
    import speech_recognition as sr
    language = request.args.get('lang', 'en')
    r = sr.Recognizer()
    global history
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        try:
            audio = r.listen(source, timeout=7, phrase_time_limit=7)
            text = r.recognize_google(audio)
            print("📝 You said:", text)
            # Store the transcribed text in the session and redirect
            #session['user_voice_message'] = text
            res = chat1(text)
            history.append({"User": text, "AI": res})
            return(render_template('chat.html', history = history, language=language, lang=LANGUAGE_CONTENT[language]))
        except Exception as e:
            print(f"❌ Error recognizing speech: {e}")
            session['user_voice_message'] = "Could not understand audio."
    
    return redirect(url_for('chat'))

# ... (All your other routes like dashboard, iot, market, etc.) ...


# -------------------- Main Program --------------------
# if __name__ == "__main__":
    # while True:
    #     # Ask user for input mode
    #     print("\nChoose input mode:")
    #     print("1️⃣  Type your query")
    #     print("2️⃣  Speak your query")
    #     print("3️⃣  Exit")
    #     choice = input("Enter 1, 2, or 3: ").strip()

    #     if choice == "1":
    #         user_input = input("Type your farming query: ").strip()
    #     # elif choice == "2":
    #     #     user_input = get_voice_input(duration=7).strip()
    #     #     if not user_input:
    #     #         print("No valid voice input captured. Try again.")
    #     #         continue
    #     elif choice == "3":
    #         print("Exiting. Goodbye! 👋")
    #         break
    #     else:
    #         print("Invalid choice. Please select 1, 2, or 3.")
    #         continue

    #     # Run agentic system
    #     final_output = run_agentic_system(user_input)

    #     print("\n=== Final Output ===")
    #     print(final_output)
    #     print(f"User message: {user_message}")
    #     flash(f"Echo: {user_message}")
    # return (render_template('chat.html'))



@app.route('/see_history',methods = ['GET', 'POST'])
def see_history():
    if(request.method == 'POST'):
        global history
        history = []
    language = request.args.get('lang', 'en')
    return(render_template('chat.html', language=language, lang=LANGUAGE_CONTENT[language]))

@app.route('/refresh-data')
def refresh_data():
    return redirect(url_for('dashboard'))

@app.route('/test-lang')
def test_language():
    language = request.args.get('lang', 'en')
    return f"Language test: {language} - Content: {LANGUAGE_CONTENT[language]['nav_dashboard']}"



@app.route('/disease')
def disease():
    language = request.args.get('lang', 'en')  # Get language from URL parameter
    return render_template('disease.html', 
                           language=language,
                           lang=LANGUAGE_CONTENT[language])

@app.route('/disease-upload', methods=['POST'])
def disease_upload():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('disease'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('disease'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename and save file
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process with Kindwise API for disease detection
        try:
            # Get optional parameters from form
            plant_id = request.form.get('plant_id')
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')
            
            # Convert to appropriate types
            if latitude:
                latitude = float(latitude)
            if longitude:
                longitude = float(longitude)
            
            # Analyze with Kindwise API
            diagnosis_result = analyze_with_kindwise(filepath, plant_id, latitude, longitude)
            
            if not diagnosis_result:
                raise Exception("Failed to get analysis results from Kindwise API")
            
                    
        except Exception as e:
            print(f"Error processing image with Kindwise API: {e}")
            flash(f"Error analyzing image: {e}")
            # Use mock data as fallback in case of an error
            diagnosis_result = {
                'disease': 'Early Blight (Fallback)',
                'scientific_name': 'Alternaria solani',
                'severity': 'High',
                'confidence': '92%',
                'treatment': 'Apply copper-based fungicide, remove infected leaves, improve air circulation',
                'prevention': 'Crop rotation, proper spacing, avoid overhead watering, remove plant debris',
                'plant_name': 'Unknown Plant',
                'plant_scientific_name': 'Unknown'
            }
        
        return render_template('disease.html', 
                               uploaded_image=filename,
                               diagnosis_result=diagnosis_result)
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('disease'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

















