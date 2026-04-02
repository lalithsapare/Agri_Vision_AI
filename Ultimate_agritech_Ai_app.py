import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import joblib

st.set_page_config(page_title='Agrivision Ai', page_icon='🌾', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
.main {background: linear-gradient(135deg, #f0fdf4 0%, #ecfccb 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #14532d 0%, #166534 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero {background: linear-gradient(90deg, #166534, #22c55e); color: white; padding: 1.5rem; border-radius: 20px; margin-bottom: 1rem; box-shadow: 0 12px 28px rgba(22,101,52,.25);}
.card {background: white; border-radius: 18px; padding: 1rem; border: 1px solid #dcfce7; box-shadow: 0 8px 18px rgba(0,0,0,.08);}
.metric-card {background: white; border-radius: 18px; padding: 1rem; text-align: center; border: 1px solid #dcfce7; box-shadow: 0 8px 18px rgba(0,0,0,.08);}
.small {color: #64748b; font-size: .9rem;}
.good {background:#dcfce7;color:#166534;padding:.3rem .7rem;border-radius:999px;font-weight:700;font-size:.8rem;}
.warn {background:#fef3c7;color:#92400e;padding:.3rem .7rem;border-radius:999px;font-weight:700;font-size:.8rem;}
.bad {background:#fee2e2;color:#b91c1c;padding:.3rem .7rem;border-radius:999px;font-weight:700;font-size:.8rem;}
</style>
''', unsafe_allow_html=True)

try:
    from google import genai
    from google.genai import types
    GENAI_MODE = 'new'
    GENAI_IMPORT_OK = True
except Exception:
    try:
        import google.generativeai as legacy_genai
        GENAI_MODE = 'legacy'
        GENAI_IMPORT_OK = True
    except Exception as e:
        GENAI_MODE = 'none'
        GENAI_IMPORT_OK = False
        GENAI_IMPORT_ERROR = str(e)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'latest_farm_data' not in st.session_state:
    st.session_state.latest_farm_data = None
if 'district' not in st.session_state:
    st.session_state.district = 'Hyderabad'
if 'season' not in st.session_state:
    st.session_state.season = 'Kharif'
if 'language' not in st.session_state:
    st.session_state.language = 'English'


def get_gemini_api_key():
    try:
        if 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY'].strip()
    except Exception:
        pass
    for key_name in ['GEMINI_API_KEY', 'GOOGLE_API_KEY']:
        value = os.getenv(key_name, '').strip()
        if value:
            return value
    return ''


def get_ai_backend_status():
    api_key = get_gemini_api_key()
    if not api_key:
        return False, 'API key missing. Add GEMINI_API_KEY in Streamlit secrets.'
    if not GENAI_IMPORT_OK:
        return False, f'Gemini SDK import failed: {GENAI_IMPORT_ERROR}'
    if GENAI_MODE == 'new':
        return True, 'Using new google-genai SDK.'
    if GENAI_MODE == 'legacy':
        return True, 'Using legacy google-generativeai SDK fallback.'
    return False, 'No Gemini SDK available.'


def generate_ai_reply(user_text, farm_data, district, season, language):
    api_key = get_gemini_api_key()
    if not api_key:
        return 'Gemini API key missing. Add GEMINI_API_KEY in Streamlit secrets.'

    farm_context = 'No latest farm analysis available.'
    if farm_data:
        farm_context = (
            f"District: {district}. Season: {season}. Language: {language}. "
            f"Crop: {farm_data.get('crop')}. Irrigation: {farm_data.get('irrigation')} ({farm_data.get('action')}). "
            f"Yield: {farm_data.get('yield')} t/ha. NDVI: {farm_data.get('ndvi')}. Health: {farm_data.get('health')}%. "
            f"Fertilizer: {farm_data.get('fertilizer')}."
        )

    prompt = f'''
You are Agrivision Ai, an expert agriculture AI assistant for Telangana farmers and agri students.
Use simple beginner-friendly language.
Reply in {language}.
Structure:
1. Current issue
2. Why it happens
3. Immediate action
4. Next 3 days plan
5. Extra agronomy tip

Farm context:
{farm_context}

User question:
{user_text}
'''
    try:
        if GENAI_MODE == 'new':
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=600)
            )
            return response.text if getattr(response, 'text', None) else 'No response generated.'
        elif GENAI_MODE == 'legacy':
            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text if getattr(response, 'text', None) else 'No response generated.'
        else:
            return 'No Gemini SDK installed.'
    except Exception as e:
        return f'AI reply error: {e}'


class DemoAgriModels:
    def __init__(self):
        self.crops = {
            'Kharif': ['Cotton', 'Maize', 'Red Gram', 'Soybean', 'Turmeric', 'Jowar', 'Paddy'],
            'Rabi': ['Bengal Gram', 'Groundnut', 'Sesame', 'Black Gram', 'Jowar', 'Paddy']
        }

    def predict_crop_recommendation(self, features, season='Kharif'):
        season_crops = self.crops.get(season, self.crops['Kharif'])
        idx = int(sum(features)) % len(season_crops)
        return season_crops[idx], round(86 + (idx % 9), 1)

    def predict_crop_yield(self, features):
        ph, temp, rainfall, fertilizer, humidity, soil_moisture = features
        value = (temp * 0.55) + (rainfall * 0.025) + (fertilizer * 0.04) + (humidity * 0.08) + (soil_moisture * 0.22) - abs(ph - 6.8) * 2
        return round(max(0.5, value), 2), 89.0

    def predict_irrigation(self, features):
        soil_moisture, temp, humidity, ph, rainfall = features
        if rainfall > 150 or soil_moisture > 55:
            return 'Low', 'Skip irrigation for now', 92.0
        if soil_moisture < 28 and temp > 32:
            return 'High', 'Give deep irrigation today', 91.0
        return 'Moderate', 'Give light irrigation', 90.0

    def calculate_ndvi(self, red, nir):
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_health_score(self, ndvi, moisture, temp):
        score = (ndvi * 55) + (moisture * 0.35) + ((35 - abs(temp - 28)) * 0.25)
        return round(min(100, max(0, score)), 1)

    def predict_fertilizer(self, n, p, k):
        if n < 50:
            return 'Apply nitrogen fertilizer in split doses', 87.0
        if p < 40:
            return 'Apply phosphorus in basal dose', 86.0
        if k < 40:
            return 'Apply potash for stress tolerance', 85.0
        return 'Use balanced NPK with FYM or compost', 89.0

    def predict_npk(self, ph, ec, organic_carbon, moisture, temperature, rainfall):
        n = round(45 + moisture * 0.4 + organic_carbon * 20 - abs(ph - 7) * 3, 2)
        p = round(25 + organic_carbon * 10 + rainfall * 0.02, 2)
        k = round(30 + temperature * 0.5 + ec * 8, 2)
        return {'N_pred': max(0, n), 'P_pred': max(0, p), 'K_pred': max(0, k)}

    def predict_price(self, year, month, market_code, arrival_qty, demand_index, crop_code):
        value = 1800 + (month * 25) + (demand_index * 120) - (arrival_qty * 0.4) + (crop_code * 10)
        return round(max(1000, value), 2)

    def predict_harvest_time(self, days_after_sowing, temperature, humidity, ph, rainfall, soil_moisture):
        maturity_gap = max(5, 120 - days_after_sowing + abs(28 - temperature) - soil_moisture * 0.1)
        return round(maturity_gap, 1)

    def predict_temperature(self, sensor_1, sensor_2):
        return round((sensor_1 + sensor_2) / 2, 2)

    def predict_rainfall(self, temperature, humidity, pressure, wind_speed):
        return round(max(0, humidity * 1.7 - pressure * 0.05 + wind_speed * 0.8 + temperature * 0.4), 2)

    def predict_humidity(self, temperature, pressure, wind_speed):
        return round(max(20, min(100, 95 - temperature * 0.8 + wind_speed * 0.4 - (pressure - 1000) * 0.2)), 2)

    def predict_ph(self, soil_moisture, organic_matter, temperature, rainfall):
        return round(max(4.5, min(8.5, 6.5 + organic_matter * 0.15 + soil_moisture * 0.01 - rainfall * 0.001)), 2)

    def predict_stress(self, ndvi, temperature, soil_moisture, humidity):
        risk = (0.4 - ndvi) * 100 + max(0, temperature - 32) * 2 + max(0, 30 - soil_moisture) + max(0, 45 - humidity)
        if risk > 35:
            return 'High Stress'
        if risk > 18:
            return 'Moderate Stress'
        return 'Low Stress'

    def predict_image_label(self, image_array, labels):
        idx = int(np.mean(image_array)) % len(labels)
        return {'class_index': idx, 'class_name': labels[idx], 'confidence': 0.88}


MODELS = DemoAgriModels()


def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr


def render_header():
    st.markdown(f"""
    <div class='hero'>
        <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
            <div>
                <h1 style='margin:0;font-size:2.2rem;'>Agrivision Ai</h1>
                <p style='margin:.35rem 0 0 0;'>Ultimate Agritech AI Module</p>
            </div>
            <div style='text-align:right;'>
                <div><span class='good'>Telangana Edition</span></div>
                <div style='margin-top:.45rem;'>📍 {st.session_state.district} | 🌾 {st.session_state.season} | 🗣️ {st.session_state.language}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def dashboard_cards():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div class='small'>Overall Health</div><h2 style='color:#16a34a;'>87.4%</h2><div class='small'>+3.2% weekly</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='small'>Active Alerts</div><h2 style='color:#dc2626;'>4</h2><div class='small'>2 critical zones</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='small'>Yield Forecast</div><h2 style='color:#166534;'>4.2 t/ha</h2><div class='small'>On target</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='small'>Water Status</div><h2 style='color:#2563eb;'>62%</h2><div class='small'>Optimal</div></div>", unsafe_allow_html=True)


st.sidebar.title('🌾 Agrivision Ai')
st.session_state.district = st.sidebar.selectbox('District', ['Hyderabad', 'Ranga Reddy', 'Medchal', 'Sangareddy', 'Mahabubnagar'])
st.session_state.season = st.sidebar.selectbox('Season', ['Kharif', 'Rabi'])
st.session_state.language = st.sidebar.selectbox('Language', ['English', 'Telugu'])

ok, msg = get_ai_backend_status()
st.sidebar.markdown('### Gemini Status')
if ok:
    st.sidebar.success(msg)
else:
    st.sidebar.error(msg)

st.sidebar.markdown('### SDK Mode')
if GENAI_MODE == 'new':
    st.sidebar.markdown("<span class='good'>google-genai</span>", unsafe_allow_html=True)
elif GENAI_MODE == 'legacy':
    st.sidebar.markdown("<span class='warn'>google-generativeai fallback</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span class='bad'>No SDK</span>", unsafe_allow_html=True)

page = st.sidebar.radio('Choose Module', [
    'Dashboard',
    'Smart Advisor',
    'Crop Recommendation',
    'Yield Prediction',
    'Irrigation',
    'Fertilizer & Soil',
    'NDVI Analysis',
    'NPK Prediction',
    'Stress Detection',
    'Price Forecast',
    'Harvest Time',
    'Weather Lab',
    'Disease Detection',
    'Leaf Classification',
    'Weed Detection',
    'Soil Classification',
    'Nutrient Deficiency',
    'AI Assistant'
])

render_header()
dashboard_cards()

if page == 'Dashboard':
    st.markdown('## Dashboard')
    trend_df = pd.DataFrame({'Day': [f'Day {i}' for i in range(1, 31)], 'Health Index': [72,74,73,75,77,76,78,79,80,78,81,83,82,84,85,86,84,87,88,89,87,90,91,89,92,93,91,94,95,96]})
    zone_df = pd.DataFrame({'Zone': ['Healthy', 'Moderate', 'High Risk', 'Critical'], 'Value': [52, 28, 14, 6]})
    col1, col2 = st.columns([2,1])
    with col1:
        fig = px.line(trend_df, x='Day', y='Health Index', title='Crop Health Trend - 30 Days')
        fig.update_traces(line=dict(color='#16a34a', width=4))
        fig.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        pie = px.pie(zone_df, names='Zone', values='Value', hole=.55, title='Zone Distribution')
        pie.update_layout(template='plotly_white', height=350)
        st.plotly_chart(pie, use_container_width=True)

elif page == 'Smart Advisor':
    st.markdown('## Smart Advisor')
    a, b, c = st.columns(3)
    with a:
        n = st.number_input('Nitrogen (N)', value=88.0)
        p = st.number_input('Phosphorus (P)', value=40.0)
        k = st.number_input('Potassium (K)', value=41.0)
        ph = st.number_input('Soil pH', value=6.7)
    with b:
        temp = st.number_input('Temperature (°C)', value=29.0)
        humidity = st.number_input('Humidity (%)', value=78.0)
        rainfall = st.number_input('Rainfall (mm)', value=180.0)
    with c:
        moisture = st.number_input('Soil Moisture (%)', value=42.0)
        red = st.number_input('Red Band', min_value=0.0, max_value=1.0, value=0.30)
        nir = st.number_input('NIR Band', min_value=0.0, max_value=1.0, value=0.72)
    if st.button('Run Complete Analysis', type='primary', use_container_width=True):
        crop, crop_conf = MODELS.predict_crop_recommendation([n, p, k, temp, humidity, ph, rainfall], st.session_state.season)
        irrigation, action, irr_conf = MODELS.predict_irrigation([moisture, temp, humidity, ph, rainfall])
        yield_pred, yield_conf = MODELS.predict_crop_yield([ph, temp, rainfall, n, humidity, moisture])
        ndvi = MODELS.calculate_ndvi(red, nir)
        health = MODELS.predict_health_score(ndvi, moisture, temp)
        fert, fert_conf = MODELS.predict_fertilizer(n, p, k)
        st.session_state.latest_farm_data = {'crop': crop, 'irrigation': irrigation, 'action': action, 'yield': yield_pred, 'ndvi': ndvi, 'health': health, 'fertilizer': fert}
        x1, x2, x3 = st.columns(3)
        x1.metric('Crop', crop, f'{crop_conf}% confidence')
        x2.metric('Irrigation', irrigation, action)
        x3.metric('Yield', f'{yield_pred} t/ha', f'{yield_conf}% confidence')
        st.markdown(f"<div class='card'><h3>Farm Decision</h3><p><b>Recommended crop:</b> {crop}</p><p><b>Irrigation:</b> {irrigation} - {action}</p><p><b>Fertilizer:</b> {fert}</p><p><b>NDVI:</b> {ndvi} | <b>Health score:</b> {health}%</p><p><b>Expected yield:</b> {yield_pred} t/ha</p></div>", unsafe_allow_html=True)

elif page == 'Crop Recommendation':
    st.markdown('## Crop Recommendation')
    vals = [st.number_input(name, value=val) for name, val in zip(['N','P','K','temperature','humidity','ph','rainfall'], [80.0,40.0,40.0,28.0,75.0,6.8,160.0])]
    if st.button('Recommend Crop', use_container_width=True):
        crop, conf = MODELS.predict_crop_recommendation(vals, st.session_state.season)
        st.success(f'Recommended crop: {crop} ({conf}% confidence)')

elif page == 'Yield Prediction':
    st.markdown('## Yield Prediction')
    vals = [st.number_input(name, value=val) for name, val in zip(['ph','temperature','rainfall','fertilizer','humidity','soil_moisture'], [6.8,28.0,150.0,90.0,76.0,40.0])]
    if st.button('Predict Yield', use_container_width=True):
        y, conf = MODELS.predict_crop_yield(vals)
        st.metric('Expected Yield', f'{y} t/ha', f'{conf}% confidence')

elif page == 'Irrigation':
    st.markdown('## Irrigation')
    vals = [st.number_input(name, value=val) for name, val in zip(['soil_moisture','temperature','humidity','ph','rainfall'], [35.0,31.0,70.0,6.8,90.0])]
    if st.button('Get Irrigation Plan', use_container_width=True):
        result, action, conf = MODELS.predict_irrigation(vals)
        st.info(f'Need: {result} | Action: {action} | Confidence: {conf}%')

elif page == 'Fertilizer & Soil':
    st.markdown('## Fertilizer & Soil')
    n = st.number_input('Nitrogen', value=45.0)
    p = st.number_input('Phosphorus', value=38.0)
    k = st.number_input('Potassium', value=42.0)
    if st.button('Analyze Soil', use_container_width=True):
        fert, conf = MODELS.predict_fertilizer(n, p, k)
        df = pd.DataFrame({'Nutrient':['N','P','K'], 'Value':[n,p,k]})
        fig = px.bar(df, x='Nutrient', y='Value', color='Nutrient', title='Soil Nutrient Levels')
        st.plotly_chart(fig, use_container_width=True)
        st.success(f'Recommendation: {fert} ({conf}% confidence)')

elif page == 'NDVI Analysis':
    st.markdown('## NDVI Analysis')
    red = st.slider('Red Band', 0.0, 1.0, 0.30)
    nir = st.slider('NIR Band', 0.0, 1.0, 0.72)
    temp = st.slider('Temperature', 15.0, 45.0, 29.0)
    moisture = st.slider('Soil Moisture', 10.0, 80.0, 42.0)
    if st.button('Calculate NDVI', use_container_width=True):
        ndvi = MODELS.calculate_ndvi(red, nir)
        health = MODELS.predict_health_score(ndvi, moisture, temp)
        gauge = go.Figure(go.Indicator(mode='gauge+number', value=health, title={'text':'Health Score'}, gauge={'axis':{'range':[0,100]}}))
        gauge.update_layout(height=350)
        st.plotly_chart(gauge, use_container_width=True)
        st.success(f'NDVI: {ndvi} | Health score: {health}%')

elif page == 'NPK Prediction':
    st.markdown('## NPK Prediction')
    vals = [st.number_input(name, value=val) for name, val in zip(['ph','ec','organic_carbon','moisture','temperature','rainfall'], [6.8,0.6,0.8,40.0,28.0,150.0])]
    if st.button('Predict NPK', use_container_width=True):
        result = MODELS.predict_npk(*vals)
        st.json(result)

elif page == 'Stress Detection':
    st.markdown('## Stress Detection')
    vals = [st.number_input(name, value=val) for name, val in zip(['ndvi','temperature','soil_moisture','humidity'], [0.42,32.0,28.0,50.0])]
    if st.button('Detect Stress', use_container_width=True):
        stress = MODELS.predict_stress(*vals)
        st.warning(f'Predicted crop stress: {stress}')

elif page == 'Price Forecast':
    st.markdown('## Price Forecast')
    vals = [st.number_input(name, value=val) for name, val in zip(['year','month','market_code','arrival_qty','demand_index','crop_code'], [2026,4,101,1200.0,7.5,12])]
    if st.button('Forecast Price', use_container_width=True):
        price = MODELS.predict_price(*vals)
        st.metric('Predicted Price', f'₹ {price}/quintal')

elif page == 'Harvest Time':
    st.markdown('## Harvest Time')
    vals = [st.number_input(name, value=val) for name, val in zip(['days_after_sowing','temperature','humidity','ph','rainfall','soil_moisture'], [85.0,29.0,74.0,6.8,120.0,40.0])]
    if st.button('Estimate Harvest Time', use_container_width=True):
        days = MODELS.predict_harvest_time(*vals)
        st.info(f'Estimated days left for harvest: {days}')

elif page == 'Weather Lab':
    st.markdown('## Weather Lab')
    t1, t2 = st.columns(2)
    with t1:
        sensor_1 = st.number_input('Sensor 1 temperature', value=28.4)
        sensor_2 = st.number_input('Sensor 2 temperature', value=29.1)
        if st.button('Predict Temperature', use_container_width=True):
            st.metric('Predicted Temperature', f"{MODELS.predict_temperature(sensor_1, sensor_2)} °C")
    with t2:
        temperature = st.number_input('Temperature', value=30.0)
        humidity = st.number_input('Humidity', value=72.0)
        pressure = st.number_input('Pressure', value=1008.0)
        wind_speed = st.number_input('Wind Speed', value=11.0)
        if st.button('Predict Rainfall & Humidity', use_container_width=True):
            rain = MODELS.predict_rainfall(temperature, humidity, pressure, wind_speed)
            hum = MODELS.predict_humidity(temperature, pressure, wind_speed)
            st.write(f'Predicted Rainfall: {rain} mm')
            st.write(f'Predicted Humidity: {hum}%')

elif page == 'Disease Detection':
    st.markdown('## Disease Detection')
    uploaded = st.file_uploader('Upload crop leaf image', type=['jpg','jpeg','png'], key='disease')
    if uploaded:
        image, arr = process_image(uploaded)
        st.image(image, caption='Uploaded leaf image', use_container_width=True)
        if st.button('Analyze Disease', use_container_width=True):
            result = MODELS.predict_image_label(arr, ['Healthy', 'Leaf Blight', 'Rust', 'Leaf Spot'])
            st.warning(f"Predicted condition: {result['class_name']} (confidence: {round(result['confidence']*100, 2)}%)")

elif page == 'Leaf Classification':
    st.markdown('## Leaf Classification')
    uploaded = st.file_uploader('Upload leaf image', type=['jpg','jpeg','png'], key='leaf')
    if uploaded:
        image, arr = process_image(uploaded)
        st.image(image, caption='Uploaded leaf image', use_container_width=True)
        if st.button('Classify Leaf', use_container_width=True):
            result = MODELS.predict_image_label(arr, ['Cotton Leaf', 'Maize Leaf', 'Paddy Leaf', 'Tomato Leaf'])
            st.success(f"Predicted leaf: {result['class_name']} (confidence: {round(result['confidence']*100, 2)}%)")

elif page == 'Weed Detection':
    st.markdown('## Weed Detection')
    uploaded = st.file_uploader('Upload field image', type=['jpg','jpeg','png'], key='weed')
    if uploaded:
        image, arr = process_image(uploaded)
        st.image(image, caption='Uploaded field image', use_container_width=True)
        if st.button('Detect Weed', use_container_width=True):
            result = MODELS.predict_image_label(arr, ['No Weed', 'Broadleaf Weed', 'Grass Weed', 'Mixed Weed'])
            st.warning(f"Predicted weed class: {result['class_name']} (confidence: {round(result['confidence']*100, 2)}%)")

elif page == 'Soil Classification':
    st.markdown('## Soil Classification')
    uploaded = st.file_uploader('Upload soil image', type=['jpg','jpeg','png'], key='soil')
    if uploaded:
        image, arr = process_image(uploaded)
        st.image(image, caption='Uploaded soil image', use_container_width=True)
        if st.button('Classify Soil', use_container_width=True):
            result = MODELS.predict_image_label(arr, ['Black Soil', 'Red Soil', 'Alluvial Soil', 'Sandy Soil'])
            st.success(f"Predicted soil: {result['class_name']} (confidence: {round(result['confidence']*100, 2)}%)")

elif page == 'Nutrient Deficiency':
    st.markdown('## Nutrient Deficiency')
    uploaded = st.file_uploader('Upload plant image', type=['jpg','jpeg','png'], key='nutrient')
    if uploaded:
        image, arr = process_image(uploaded)
        st.image(image, caption='Uploaded plant image', use_container_width=True)
        if st.button('Analyze Deficiency', use_container_width=True):
            result = MODELS.predict_image_label(arr, ['Healthy', 'Nitrogen Deficiency', 'Potassium Deficiency', 'Phosphorus Deficiency'])
            st.error(f"Predicted nutrient status: {result['class_name']} (confidence: {round(result['confidence']*100, 2)}%)")

elif page == 'AI Assistant':
    st.markdown('## AI Assistant')
    st.write('DEBUG KEY:', get_gemini_api_key())
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if prompt := st.chat_input('Ask about irrigation, crop, disease, fertilizer, yield, or Telangana farming...'):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            with st.spinner('Agrivision Ai is analyzing...'):
                reply = generate_ai_reply(prompt, st.session_state.latest_farm_data, st.session_state.district, st.session_state.season, st.session_state.language)
                st.markdown(reply)
                st.session_state.chat_history.append({'role': 'assistant', 'content': reply})

st.markdown('---')
st.markdown("<div style='text-align:center;color:#6b7280;padding:18px;font-size:14px;'>🌾 Agrivision Ai | Ultimate Agritech AI Module | Streamlit Ready</div>", unsafe_allow_html=True)
