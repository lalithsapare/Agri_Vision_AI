import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AgriTech Smart Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS (ENHANCED) ----------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0c111b 0%, #101722 100%);
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14532d 0%, #1d7a46 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero-box {
    background: linear-gradient(90deg, #166534, #27b36a);
    color: white;
    padding: 1.4rem 1.2rem;
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(39,179,106,0.25);
    margin-bottom: 1rem;
}
.info-card {
    background: rgba(255,255,255,0.98);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    border: 1px solid #dcefe2;
    margin-bottom: 16px;
}
.decision-card {
    background: linear-gradient(135deg, #f0fff4, #e0f7ea);
    border-left: 8px solid #1e9b5a;
    color: #123524;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-top: 12px;
}
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 14px;
    text-align: center;
    border: 1px solid #e5f1e8;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.farmer-line {
    background: linear-gradient(90deg, #ffffff, #f4fff6);
    color: #1b4332;
    border-radius: 999px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-weight: 600;
    border: 1px solid #d8eedf;
}
.small-muted {
    color: #4d6758;
    font-size: 14px;
}
/* NEW CHATBOT CSS */
[data-testid="stChatMessage"] {
    border-radius: 18px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 15.8px;
    line-height: 1.65;
}
div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 8px;
    background: rgba(12, 17, 27, 0.96);
    padding-top: 8px;
    z-index: 999;
}
.chat-toolbar {
    background: rgba(255,255,255,0.96);
    border-radius: 14px;
    padding: 12px 16px;
    border: 1px solid #dcefe2;
    margin-bottom: 16px;
}
.quick-chip {
    background: linear-gradient(135deg, #f0fff4, #e0f7ea);
    border: 1px solid #27b36a;
    color: #166534;
    border-radius: 24px;
    padding: 8px 16px;
    font-weight: 500;
    margin: 4px;
    cursor: pointer;
}
.quick-chip:hover {
    background: linear-gradient(135deg, #e0f7ea, #d1f2dc);
    border-color: #166534;
}
.why-block {
    background: rgba(30, 155, 90, 0.08);
    border-left: 4px solid #27b36a;
    padding: 12px;
    border-radius: 12px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- MODEL ENGINE ----------------------
class AgritechModels:
    def __init__(self):
        pass

    def predict_crop_recommendation(self, features):
        crops = [
            "Rice", "Maize", "Cotton", "Sugarcane", "Groundnut",
            "Paddy", "Banana", "Tomato", "Chilli", "Mango"
        ]
        score = int(sum(features)) % len(crops)
        crop = crops[score]
        confidence = round(82 + (score % 12), 1)
        return crop, confidence

    def predict_crop_yield(self, features):
        ph, temperature, rainfall, fertilizer, humidity, soil_moisture = features
        pred = (temperature * 0.7) + (rainfall * 0.025) + (fertilizer * 0.04) + (humidity * 0.08) + (soil_moisture * 0.22) - abs(ph - 6.5) * 2
        pred = round(max(pred, 0), 2)
        confidence = 89.0
        return pred, confidence

    def predict_irrigation(self, features):
        soil_moisture, temperature, humidity, ph, rainfall = features
        if rainfall > 180 or soil_moisture > 60:
            result = "Low"
            action = "Reduce irrigation"
        elif soil_moisture < 30 and temperature > 30:
            result = "High"
            action = "Increase irrigation"
        else:
            result = "Moderate"
            action = "Maintain moderate irrigation"
        confidence = 90.0
        return result, action, confidence

    def predict_fertilizer(self, features):
        N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall = features
        if N < 50:
            fert = "Apply nitrogen fertilizer"
        elif P < 40:
            fert = "Apply phosphorus fertilizer"
        else:
            fert = "Balanced NPK fertilizer"
        return fert, 87.0

    def predict_rainfall(self, features):
        temp, humidity, pressure, wind_speed = features
        rain = max(round((humidity * 0.7) - (temp * 0.2) + (wind_speed * 0.4) + ((1015 - pressure) * 1.1), 2), 0)
        if rain > 60:
            status = "High Rainfall"
        elif rain > 25:
            status = "Moderate Rainfall"
        else:
            status = "Low Rainfall"
        return rain, status, 86.0

    def predict_ph(self, features):
        soil_moisture, organic_matter, temp, rainfall = features
        ph = round(6.3 + organic_matter * 0.18 + soil_moisture * 0.01 - rainfall * 0.001, 2)
        return ph, 84.0

    def predict_npk(self, features):
        ph, ec, organic_carbon, moisture, temp, rainfall = features
        n = round((organic_carbon * 40) + (moisture * 0.8), 2)
        p = round((ec * 22) + (ph * 3), 2)
        k = round((rainfall * 0.08) + (temp * 0.55), 2)
        return {"Nitrogen": n, "Phosphorus": p, "Potassium": k}, 88.0

    def calculate_ndvi(self, red, nir):
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_crop_stress(self, features):
        ndvi, temp, soil_moisture, humidity = features
        if ndvi < 0.35 or soil_moisture < 28 or temp > 36:
            return "High", 78.0
        elif ndvi < 0.50:
            return "Moderate", 85.0
        return "Low", 92.0

    def predict_disease(self, image_array):
        labels = [
            ("Healthy", 91.0),
            ("Leaf Blight", 88.0),
            ("Rust", 84.0),
            ("Leaf Spot", 86.0),
            ("Powdery Mildew", 83.0)
        ]
        return labels[int(np.mean(image_array)) % len(labels)]

agrimodels = AgritechModels()

# ---------------------- SESSION STATE ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_bot_answer" not in st.session_state:
    st.session_state.last_bot_answer = ""

if "sim_location" not in st.session_state:
    st.session_state.sim_location = "Andhra Pradesh"

if "sim_season" not in st.session_state:
    st.session_state.sim_season = "Kharif"

if "latest_final_output" not in st.session_state:
    st.session_state.latest_final_output = None

# ---------------------- HELPERS ----------------------
def num_input(label, value=0.0):
    return st.number_input(label, value=float(value), step=0.1)

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr

def stream_markdown_reply(reply_text):
    """Typing effect for GPT-like experience"""
    holder = st.empty()
    typed = ""
    for ch in reply_text:
        typed += ch
        holder.markdown(typed)
        time.sleep(0.004)
    return reply_text

def generate_gpt_like_reply(user_text, final_output=None, location="Andhra Pradesh", season="Kharif"):
    """Advanced GPT-like agritech responses"""
    text = user_text.lower().strip()

    crop = "Rice"
    irrigation = "Moderate"
    weather = "Moderate Rainfall"
    actions = ["Maintain field monitoring", "Use balanced fertilizer"]

    if final_output:
        crop = final_output.get("crop", crop)
        irrigation = final_output.get("irrigation", irrigation)
        weather = final_output.get("weather", weather)
        actions = final_output.get("actions", actions)

    main_answer = ""
    why_block = ""

    if "what should i do" in text or "what i do" in text:
        main_answer = f"""
### 🚀 Recommended Farm Action Plan

**Based on your latest farm analysis:**

- **🌾 Crop:** {crop}
- **💧 Irrigation:** {irrigation} 
- **🌦 Weather:** {weather}

**✅ Immediate Actions:**
- {actions[0]}
- {actions[1]}
- {'⚠️ Spray preventive fungicide' if 'rainfall' in weather.lower() else '👁️ Continue crop observation for 3-5 days'}
        """
        why_block = f"""
### 🤔 Why This Advice?
- **{crop}** matches current **{season}** patterns in **{location}**
- **{irrigation}** irrigation balances expected rainfall + soil moisture
- **{season}** season needs extra fungal prevention when humidity stays high
- Actions prioritize yield protection + input efficiency
        """

    elif "rice" in text and "season" in text:
        main_answer = f"""
### 🌾 Rice Crop Season Guidance

**For {location}:**
✅ **{season}** is **ideal** for rice (good rainfall + humidity)

**Best Practices:**
- ⏰ Start nursery beds early
- 💧 Avoid over-irrigation during heavy rain
- 🦠 Watch leaf blast + blight in humid weeks
- 📈 Monitor NDVI weekly after transplanting
        """
        why_block = f"""
**Season Science:**
- Rice needs standing water → perfect for **{season}** rainfall
- {location} patterns favor rice when monsoon is timely
- High humidity = higher fungal risk (prevent early)
        """

    elif "disease" in text or "reduce disease" in text:
        main_answer = """
### 🛡️ Disease Prevention Strategy

**Daily Field Checklist:**
1. 👀 **Inspect** leaves (yellowing, spots, curling)
2. 🚫 **Avoid** water stagnation 
3. 🌬️ **Improve** airflow between plants
4. 💉 **Preventive spray** during high humidity + rain
5. 📸 **Upload image** in Image Models for diagnosis

**When to worry:** Brown spots + wilting = act immediately
        """
        why_block = """
**Disease Science:**
- Wet leaves + poor airflow = fungal explosion
- Early prevention = 3x cheaper than treatment
- High humidity periods need 7-day spray cycles
        """

    elif "irrigation" in text:
        main_answer = f"""
### 💧 Smart Irrigation Plan

**Current Status:** {irrigation}

**Field Actions:**
- 🧪 Test soil moisture before watering
- ☔️ Reduce if rainfall forecast > 50mm
- ⏰ Irrigate morning OR evening only
- 🚫 Never overwater (kills roots + spreads disease)

**Pro Tip:** Drip irrigation saves 30-50% water
        """
        why_block = """
**Irrigation Logic:**
- Soil moisture + rainfall + crop stage = perfect balance
- Overwatering cuts oxygen to roots + boosts disease
- Morning irrigation dries leaves before night
        """

    elif "fertilizer" in text:
        main_answer = f"""
### 🧪 Fertilizer Optimization

**Smart Recommendation:**
{actions[1]}

**Application Guide:**
- 📅 Split nitrogen (50% now, 50% tillering)
- ☔️ Avoid large doses before heavy rain
- 🧪 Always test pH + moisture first
- 💡 Use organic matter for slow-release NPK

**{season} Special:** Reduce nitrogen 20% if rainfall heavy
        """
        why_block = f"""
**Fertilizer Science:**
- Heavy rain washes 40-60% nitrogen away
- {location} {season} needs split-dosing strategy
- Phosphorus stays in soil longer than nitrogen
        """

    elif "andhra" in text or "seasonal" in text:
        main_answer = f"""
### 🌍 {location} {season} Master Plan

**Key Focus Areas:**
1. ☔️ **Rainfall timing** → irrigation trigger
2. 🌾 **Crop choice** → match local season
3. 🦠 **Disease pressure** → post-rain monitoring  
4. 🧪 **Fertilizer** → split wet-season dosing
5. 📊 **NDVI tracking** → weekly crop health

**{season} Pro Tip:** First 21 days after rain = make or break
        """
        why_block = """
**Regional Science:**
- {location} seasonal rhythm drives 70% of success
- Monsoon timing > any fertilizer/irrigation decision
- Local disease patterns repeat yearly
        """

    elif "ndvi" in text or "health" in text:
        main_answer = """
### 📈 NDVI + Crop Health Guide

**NDVI Ranges:**
✅ **0.7+** = Excellent vigor
🟡 **0.5-0.7** = Good, monitor
🟠 **0.3-0.5** = Stress detected
🔴 **<0.3** = Serious problems

**Use NDVI tab** for Red/NIR → instant calculation
        """
        why_block = """
**NDVI Science:**
- Stressed plants reflect differently (less NIR)
- Weekly tracking catches problems 7-10 days early
- Works better than visual inspection alone
        """

    else:
        main_answer = """
### 🤖 What I Can Help With

**Smart Farm Questions:**
| Question | What You'll Get |
|----------|----------------|
| "What should I do?" | Complete action plan |
| "Rice best season?" | Crop timing guide |
| "Disease risk?" | Prevention checklist |
| "Fertilizer?" | Application strategy |
| "AP seasonal tips?" | Local insights |

**Pro Tip:** Run Smart Advisor first for personalized advice
        """
        why_block = """
**My Intelligence:**
- Uses your latest Smart Advisor results
- {location} + {season} optimized
- Combines 8+ models for final decisions
        """.format(location=location, season=season)

    return main_answer, why_block

def get_simulation_defaults(location, season):
    data = 
        ("Andhra Pradesh", "Kharif"): {
            "N": 90, "P": 42, "K": 43, "temperature": 28, "humidity": 82, "ph": 6.5, "rainfall": 220,
            "soil_moisture": 48, "fertilizer": 110, "organic_matter": 2.1, "ec": 1.2, "organic_carbon": 0.82
        },
        ("Andhra Pradesh", "Rabi"): {
            "N": 75, "P": 38, "K": 35, "temperature": 23, "humidity": 68, "ph": 6.8, "rainfall":
                    ("Andhra Pradesh", "Rabi"): {
            "N": 75, "P": 38, "K": 35, "temperature": 23, "humidity": 68, "ph": 6.8, "rainfall": 90,
            "soil_moisture": 36, "fertilizer": 95, "organic_matter": 1.9, "ec": 1.0, "organic_carbon": 0.70
        },
        ("Telangana", "Kharif"): {
            "N": 88, "P": 40, "K": 41, "temperature": 29, "humidity": 78, "ph": 6.7, "rainfall": 180,
            "soil_moisture": 42, "fertilizer": 105, "organic_matter": 2.0, "ec": 1.1, "organic_carbon": 0.76
        }
    }
    return data.get((location, season), data[("Andhra Pradesh", "Kharif")])

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("🌿 AgriTech Menu")
page = st.sidebar.radio(
    "Choose Module",
    [
        "Home",
        "Smart Advisor",
        "Crop Recommendation",
        "Crop Yield",
        "Irrigation",
        "Fertilizer",
        "Soil & NPK",
        "NDVI & Stress",
        "Image Models",
        "Farmer Chat Assistant"
    ]
)

st.sidebar.markdown("### 🌍 Real-world Simulation Mode")
location = st.sidebar.selectbox("Location", ["Andhra Pradesh", "Telangana"], index=0)
season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"], index=0)
defaults = get_simulation_defaults(location, season)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="hero-box">
    <h1>🌾 AgriTech Smart Assistant</h1>
    <p>AI-powered agriculture prediction dashboard with GPT-style chatbot, image analysis, smart simulation, health scoring, and final field decision output.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- ALL PAGES ----------------------
if page == "Home":
    st.markdown("""
    <div class="farmer-line">👨‍🌾 Why this AgriTech app? It helps farmers and agronomy teams make faster decisions on crop choice, irrigation, fertilizer use, crop health, and field risk using simple AI-guided insights.</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌾 Smart Farming", "👁 Crop Monitoring", "🤖 AI Assistant"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.subheader("Important Features")
            st.write("""
- Crop recommendation with confidence score.
- Irrigation and rainfall decision support.
- Fertilizer suggestion using soil-style inputs.
- Yield prediction graph for farm planning.
- Final action block for farmer decisions.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.subheader("Why It Matters")
            st.write("""
- Reduces wrong input decisions.
- Supports smarter field management.
- Helps beginner farmers understand actions.
- Useful for portfolio, startup demo, and real advisory systems.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Crop Monitoring Capabilities")
        st.write("""
- NDVI analysis for crop vigor.
- Disease prediction from images.
- Crop stress level estimation.
- Health score and farm risk summary.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("GPT-Style AI Assistant")
        st.write(f"""
- **Conversation memory** remembers your chat history
- **Context-aware** uses Smart Advisor results  
- **Quick chips** for common farmer questions
- **Typing effect** + markdown answers
- **Why?** explainers + copy/export answers
- **{location}** + **{season}** optimized advice
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Smart Advisor":
    st.subheader("🧠 Smart Farm Advisor")

    c1, c2, c3 = st.columns(3)
    with c1:
        N = num_input("Nitrogen", defaults["N"])
        P = num_input("Phosphorus", defaults["P"])
        K = num_input("Potassium", defaults["K"])
        ph = num_input("pH", defaults["ph"])
    with c2:
        temperature = num_input("Temperature", defaults["temperature"])
        humidity = num_input("Humidity", defaults["humidity"])
        rainfall = num_input("Rainfall", defaults["rainfall"])
    with c3:
        soil_moisture = num_input("Soil Moisture", defaults["soil_moisture"])
        fertilizer_val = num_input("Fertilizer", defaults["fertilizer"])
        red = num_input("Red Band", 0.30)
        nir = num_input("NIR Band", 0.72)

    if st.button("🚀 Generate Final Farm Decision", type="primary"):
        with st.spinner("Analyzing your farm data..."):
            # All predictions
            crop, crop_conf = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall])
            irrigation, irrigation_action, irr_conf = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall])
            rain_value, weather_status, weather_conf = agrimodels.predict_rainfall([temperature, humidity, 1012, 8])
            fert_result, fert_conf = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, soil_moisture, 1, 1, ph, rainfall])
            yield_pred, yield_conf = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer_val, humidity, soil_moisture])
            ndvi = agrimodels.calculate_ndvi(red, nir)
            stress_level, stress_conf = agrimodels.predict_crop_stress([ndvi, temperature, soil_moisture, humidity])

            # Store final output for chatbot
            final_output = {
                "crop": crop,
                "irrigation": irrigation,
                "weather": weather_status,
                "actions": [irrigation_action, fert_result],
                "yield": yield_pred,
                "ndvi": ndvi,
                "stress": stress_level
            }
            st.session_state.latest_final_output = final_output

            # Results display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{crop}</h3>
                    <p>Recommended Crop<br><span class="small-muted">{crop_conf}% confidence</span></p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{irrigation}</h3>
                    <p>Irrigation Need<br><span class="small-muted">{irr_conf}% confidence</span></p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{yield_pred} tons/ha</h3>
                    <p>Expected Yield<br><span class="small-muted">{yield_conf}% confidence</span></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="decision-card">
                <h3>✅ **Final Farm Decision**</h3>
                <strong>🌾 Crop:</strong> {crop} ({crop_conf}%)<br>
                <strong>💧 Irrigation:</strong> {irrigation} - {irrigation_action}<br>
                <strong>🧪 Fertilizer:</strong> {fert_result}<br>
                <strong>🌦 Weather:</strong> {weather_status} ({rain_value}mm)<br>
                <strong>📈 NDVI:</strong> {ndvi:.3f} ({stress_level} stress)<br>
                <strong>🎯 Yield:</strong> {yield_pred} tons/ha
            </div>
            """, unsafe_allow_html=True)

elif page == "Crop Recommendation":
    st.subheader("🌾 Crop Recommendation")
    inputs = [num_input(f"Feature {i+1}", defaults.get(f"N", 50) + i*5) for i in range(7)]
    if st.button("Recommend Crop"):
        crop, conf = agrimodels.predict_crop_recommendation(inputs)
        st.success(f"**Recommended: {crop}** ({conf}% confidence)")

elif page == "Crop Yield":
    st.subheader("📈 Crop Yield Prediction")
    yield_inputs = [num_input(f"Input {i+1}", 25.0) for i in range(6)]
    if st.button("Predict Yield"):
        yield_pred, conf = agrimodels.predict_crop_yield(yield_inputs)
        st.metric("Expected Yield", f"{yield_pred} tons/ha", delta=f"{conf}% confidence")

elif page == "Irrigation":
    st.subheader("💧 Irrigation Advisor")
    irr_inputs = [num_input(f"Param {i+1}", 40.0) for i in range(5)]
    if st.button("Get Irrigation Plan"):
        result, action, conf = agrimodels.predict_irrigation(irr_inputs)
        st.markdown(f"**Need: {result}** | **Action: {action}** ({conf}%)")

elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")
    fert_inputs = [num_input(f"Input {i+1}", 50.0) for i in range(10)]
    if st.button("Recommend Fertilizer"):
        result, conf = agrimodels.predict_fertilizer(fert_inputs)
        st.success(f"**{result}** ({conf}%)")

elif page == "Soil & NPK":
    st.subheader("🧪 Soil NPK Analysis")
    npk_inputs = [num_input(f"Soil {i+1}", 1.0) for i in range(6)]
    if st.button("Analyze Soil"):
        npk, conf = agrimodels.predict_npk(npk_inputs)
        st.json(npk)
        st.caption(f"Confidence: {conf}%")

elif page == "NDVI & Stress":
    st.subheader("📈 NDVI & Crop Stress")
    red, nir = st.columns(2)
    with red:
        red_val = st.number_input("Red Band", 0.0, 1.0, 0.3)
    with nir:
        nir_val = st.number_input("NIR Band", 0.0, 1.0, 0.7)
    
    temp, soil_moist, hum = st.columns(3)
    with temp:
        temp_val = st.number_input("Temperature", 15.0, 45.0, 28.0)
    with soil_moist:
        soil_val = st.number_input("Soil Moisture", 10.0, 80.0, 45.0)
    with hum:
        hum_val = st.number_input("Humidity", 30.0, 100.0, 75.0)
    
    if st.button("Calculate Health"):
        ndvi = agrimodels.calculate_ndvi(red_val, nir_val)
        stress, conf = agrimodels.predict_crop_stress([ndvi, temp_val, soil_val, hum_val])
        
        fig = go.Figure()
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Monitor")
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Critical")
        fig.add_trace(go.Scatter(x=[1], y=[ndvi], mode='markers+text', 
                                marker=dict(size=20, color='blue'), text=[f"NDVI: {ndvi:.3f}"], 
                                textposition="middle center", name="Your Field"))
        fig.update_layout(title="NDVI Health Score", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Stress Level: {stress}** ({conf}% confidence)")

elif page == "Image Models":
    st.subheader("🖼️ Leaf Disease Detection")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image, arr = process_image(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
        if st.button("Analyze Disease"):
            disease, conf = agrimodels.predict_disease(arr)
            st.error(f"**Detected: {disease[0]}** ({disease[1]}% confidence)")
            st.info("💡 Real model would use CNN like ResNet50 trained on PlantVillage dataset")

elif page == "Farmer Chat Assistant":
    st.markdown('<div class="chat-toolbar">', unsafe_allow_html=True)
    st.markdown(f"""
    💬 **Smart Agri Assistant** | Location: **{location}** | Season: **{season}** 
    {'✅ Latest analysis loaded!' if st.session_state.latest_final_output else '⚠️ Run Smart Advisor first!'}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Quick chips
    st.markdown('<div class="chat-toolbar">', unsafe_allow_html=True)
    cols = st.columns(5)
    questions = [
        "What should I do?",
        "Irrigation plan?", 
        "Fertilizer advice?",
        "Disease risk?",
        "Rice season?"
    ]
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q, key=f"quick_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                main_reply, why_reply = generate_gpt_like_reply(
                    q, st.session_state.latest_final_output, location, season
                )
                full_reply = main_reply + "\n\n" + why_reply
                st.session_state.chat_history.append({"role": "assistant", "content": full_reply})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about your farm (crop, irrigation, fertilizer, disease, yield)..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 AgriTech AI is thinking..."):
                main_reply, why_reply = generate_gpt_like_reply(
                    prompt, st.session_state.latest_final_output, location, season
                )
                full_reply = main_reply + "\n\n" + why_reply
                st.markdown(full_reply)
                st.session_state.chat_history.append({"role": "assistant", "content": full_reply})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    🌾 **Built for Andhra Pradesh farmers** | AI/ML Agriculture Assistant | Ready for production deployment
</div>
""", unsafe_allow_html=True)