from flask import Flask, render_template, request, redirect, url_for, flash
import os
import requests
from dotenv import load_dotenv
from PIL import Image
from werkzeug.utils import secure_filename
import uuid
import base64
import random
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
load_dotenv()
# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
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

# Mock data generators
def get_iot_data():
    return {
        'soil_health': {
            'ph': round(random.uniform(6.0, 7.5), 1),
            'nutrients': random.choice(['Optimal', 'Good', 'Fair']),
            'moisture': random.randint(60, 85)
        },
        'plant_health': {
            'growth': random.choice(['Healthy', 'Good', 'Moderate']),
            'stress': random.choice(['None', 'Low', 'Moderate'])
        },
        'weather_alert': {
            'type': random.choice(['Frost Advisory', 'Heavy Rain Warning', 'Heat Wave Alert']),
            'warning': random.choice([
                'Potential frost advisory in 48 hours',
                'Heavy rainfall expected in 24 hours',
                'High temperatures expected this week'
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
@app.route('/')
def dashboard():
    iot_data = get_iot_data()
    predictions = get_growth_predictions()
    
    return render_template('dashboard.html', 
                           iot_data=iot_data,
                           predictions=predictions)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('dashboard'))
    
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
        
        return render_template('dashboard.html', 
                               uploaded_image=filename,
                               diagnosis_result=diagnosis_result,
                               iot_data=get_iot_data(),
                               predictions=get_growth_predictions())
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('dashboard'))

@app.route('/iot')
def iot_monitoring():
    iot_data = get_iot_data()
    predictions = get_growth_predictions()
    crop_varieties = get_crop_varieties()
    
    return render_template('iot_monitoring_growthing.html', 
                           iot_data=iot_data,
                           predictions=predictions,
                           crop_varieties=crop_varieties)

@app.route('/market')
def market_dashboard():
    market_data = get_market_data()
    
    # Calculate price trends
    wheat_trend = {
        'change': '+2.5%',
        'direction': 'up',
        'last_30_days': '+2.5%'
    }
    
    return render_template('market_dashboard.html', 
                           market_data=market_data,
                           wheat_trend=wheat_trend)

# Disease detection is now integrated into the main dashboard

@app.route('/refresh-data')
def refresh_data():
    return redirect(url_for('dashboard'))



@app.route('/disease')
def disease():
    return render_template('disease.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
