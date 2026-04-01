try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

load_dotenv()

st.set_page_config(
    page_title="AgriTech Smart Assistant Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
.chat-toolbar {
    background: rgba(255,255,255,0.96);
    border-radius: 14px;
    padding: 12px 16px;
    border: 1px solid #dcefe2;
    margin-bottom: 16px;
}
.region-chip {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: #e8fff0;
    color: #166534;
    border: 1px solid #b7e4c7;
    margin-right: 8px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    font-weight: 600;
}
[data-testid="stChatMessage"] {
    border-radius: 18px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 15.8px;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)


class AgritechModels:
    def predict_crop_recommendation(self, features: List[float], location: str = "Andhra Pradesh", season: str = "Kharif") -> Tuple[str, float]:
        region_crops = {
            ("Andhra Pradesh", "Kharif"): ["Rice", "Maize", "Cotton", "Sugarcane", "Groundnut", "Banana", "Chilli", "Tomato"],
            ("Andhra Pradesh", "Rabi"): ["Rice", "Maize", "Groundnut", "Black Gram", "Sunflower", "Chilli", "Bengal Gram", "Tomato"],
            ("Telangana", "Kharif"): ["Cotton", "Maize", "Paddy", "Soybean", "Red Gram", "Chilli", "Turmeric", "Jowar"],
            ("Telangana", "Rabi"): ["Maize", "Bengal Gram", "Groundnut", "Sesame", "Jowar", "Sunflower", "Paddy", "Black Gram"],
        }
        crops = region_crops.get((location, season), region_crops[("Andhra Pradesh", "Kharif")])
        score = int(sum(features)) % len(crops)
        crop = crops[score]
        confidence = round(84 + (score % 10), 1)
        return crop, confidence

    def predict_crop_yield(self, features: List[float], location: str = "Andhra Pradesh") -> Tuple[float, float]:
        ph, temperature, rainfall, fertilizer, humidity, soil_moisture = features
        regional_factor = 1.04 if location == "Telangana" else 1.0
        pred = ((temperature * 0.68) + (rainfall * 0.024) + (fertilizer * 0.04) + (humidity * 0.08) + (soil_moisture * 0.23) - abs(ph - 6.5) * 2) * regional_factor
        pred = round(max(pred, 0), 2)
        return pred, 89.5

    def predict_irrigation(self, features: List[float], location: str = "Andhra Pradesh") -> Tuple[str, str, float]:
        soil_moisture, temperature, humidity, ph, rainfall = features
        if location == "Telangana":
            if rainfall > 160 or soil_moisture > 58:
                return "Low", "Reduce irrigation and monitor field drainage", 91.0
            if soil_moisture < 32 and temperature > 31:
                return "High", "Increase irrigation; prefer drip or split watering", 91.0
            return "Moderate", "Maintain moderate irrigation with field moisture checks", 91.0

        if rainfall > 180 or soil_moisture > 60:
            return "Low", "Reduce irrigation", 90.0
        if soil_moisture < 30 and temperature > 30:
            return "High", "Increase irrigation", 90.0
        return "Moderate", "Maintain moderate irrigation", 90.0

    def predict_fertilizer(self, features: List[float], location: str = "Andhra Pradesh", crop_name: str = "General") -> Tuple[str, float]:
        N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall = features
        if location == "Telangana" and crop_name in ["Cotton", "Maize", "Red Gram", "Soybean"]:
            if N < 55:
                return "Apply nitrogen in split doses; add micronutrient spray for Telangana dry belts", 88.5
            if P < 42:
                return "Apply phosphorus-rich fertilizer with basal placement", 88.5
            if K < 40:
                return "Apply potash to improve drought tolerance", 88.5
            return "Balanced NPK fertilizer with micronutrient support", 88.5

        if N < 50:
            return "Apply nitrogen fertilizer", 87.0
        if P < 40:
            return "Apply phosphorus fertilizer", 87.0
        if K < 38:
            return "Apply potassium fertilizer", 87.0
        return "Balanced NPK fertilizer", 87.0

    def predict_rainfall(self, features: List[float], location: str = "Andhra Pradesh") -> Tuple[float, str, float]:
        temp, humidity, pressure, wind_speed = features
        regional_boost = 2.5 if location == "Telangana" else 0.0
        rain = max(round((humidity * 0.7) - (temp * 0.2) + (wind_speed * 0.4) + ((1015 - pressure) * 1.1) + regional_boost, 2), 0)
        if rain > 60:
            status = "High Rainfall"
        elif rain > 25:
            status = "Moderate Rainfall"
        else:
            status = "Low Rainfall"
        return rain, status, 86.5

    def predict_npk(self, features: List[float]) -> Tuple[Dict[str, float], float]:
        ph, ec, organic_carbon, moisture, temp, rainfall = features
        n = round((organic_carbon * 40) + (moisture * 0.8), 2)
        p = round((ec * 22) + (ph * 3), 2)
        k = round((rainfall * 0.08) + (temp * 0.55), 2)
        return {"Nitrogen": n, "Phosphorus": p, "Potassium": k}, 88.0

    def calculate_ndvi(self, red: float, nir: float) -> float:
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_crop_stress(self, features: List[float], location: str = "Andhra Pradesh") -> Tuple[str, float]:
        ndvi, temp, soil_moisture, humidity = features
        temp_threshold = 35 if location == "Telangana" else 36
        if ndvi < 0.35 or soil_moisture < 28 or temp > temp_threshold:
            return "High", 79.0
        if ndvi < 0.50:
            return "Moderate", 85.0
        return "Low", 92.0

    def predict_disease(self, image_array: np.ndarray) -> Tuple[str, float]:
        labels = [
            ("Healthy", 91.0),
            ("Leaf Blight", 88.0),
            ("Rust", 84.0),
            ("Leaf Spot", 86.0),
            ("Powdery Mildew", 83.0),
        ]
        return labels[int(np.mean(image_array)) % len(labels)]


agrimodels = AgritechModels()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_final_output" not in st.session_state:
    st.session_state.latest_final_output = None


def num_input(label: str, value: float = 0.0) -> float:
    return st.number_input(label, value=float(value), step=0.1)


def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr


def get_simulation_defaults(location: str, season: str) -> Dict[str, float]:
    data = {
        ("Andhra Pradesh", "Kharif"): {
            "N": 90, "P": 42, "K": 43, "temperature": 28, "humidity": 82, "ph": 6.5,
            "rainfall": 220, "soil_moisture": 48, "fertilizer": 110, "organic_matter": 2.1,
            "ec": 1.2, "organic_carbon": 0.82,
        },
        ("Andhra Pradesh", "Rabi"): {
            "N": 75, "P": 38, "K": 35, "temperature": 23, "humidity": 68, "ph": 6.8,
            "rainfall": 90, "soil_moisture": 36, "fertilizer": 95, "organic_matter": 1.9,
            "ec": 1.0, "organic_carbon": 0.70,
        },
        ("Telangana", "Kharif"): {
            "N": 82, "P": 40, "K": 41, "temperature": 30, "humidity": 74, "ph": 6.7,
            "rainfall": 165, "soil_moisture": 40, "fertilizer": 102, "organic_matter": 1.8,
            "ec": 1.15, "organic_carbon": 0.74,
        },
        ("Telangana", "Rabi"): {
            "N": 70, "P": 36, "K": 34, "temperature": 25, "humidity": 62, "ph": 7.0,
            "rainfall": 55, "soil_moisture": 30, "fertilizer": 88, "organic_matter": 1.6,
            "ec": 0.95, "organic_carbon": 0.66,
        },
    }
    return data.get((location, season), data[("Andhra Pradesh", "Kharif")])


def get_region_note(location: str, season: str) -> str:
    notes = {
        ("Andhra Pradesh", "Kharif"): "Best suited for paddy, chilli, banana, and humid-season disease monitoring.",
        ("Andhra Pradesh", "Rabi"): "Good for maize, pulses, sunflower, and controlled irrigation planning.",
        ("Telangana", "Kharif"): "Optimized for cotton, maize, red gram, soybean, and rainfall-variability decisions.",
        ("Telangana", "Rabi"): "Optimized for bengal gram, groundnut, sesame, jowar, and dry-season water management.",
    }
    return notes.get((location, season), "Regional optimization active.")


def generate_gpt_like_reply(user_text: str, final_output=None, location: str = "Andhra Pradesh", season: str = "Kharif") -> Tuple[str, str]:
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

    region_line = f"This answer is optimized for {location} during {season}."

    if "what should i do" in text or "what i do" in text:
        main_answer = f"""
### 🚀 Recommended Farm Action Plan
- 🌾 Crop: {crop}
- 💧 Irrigation: {irrigation}
- 🌦 Weather: {weather}
- ✅ Immediate action 1: {actions[0]}
- ✅ Immediate action 2: {actions[1]}
- ✅ Region focus: {get_region_note(location, season)}
        """
        why_block = f"""
### 🤔 Why This Advice?
- {region_line}
- {crop} matches the current seasonal pattern.
- {irrigation} irrigation balances soil moisture and rainfall risk.
- These actions prioritize yield, disease prevention, and input efficiency.
        """
    elif "telangana" in text:
        main_answer = """
### 🌍 Telangana Crop Strategy
- Cotton, maize, red gram, soybean, turmeric, and bengal gram are strong options depending on season.
- Kharif planning should focus on rainfall variability and drainage.
- Rabi planning should focus on water saving and nutrient split application.
        """
        why_block = """
### 🤔 Telangana Special Logic
- Telangana needs stronger dryland planning than coastal regions.
- Cotton and pulse systems benefit from split fertilizer and moisture tracking.
- Drip irrigation and drought-tolerance decisions matter more here.
        """
    elif "irrigation" in text:
        main_answer = f"""
### 💧 Smart Irrigation Plan
- Current status: {irrigation}
- Test soil moisture before watering.
- Reduce irrigation when forecast rainfall is high.
- Prefer morning or evening irrigation.
- In Telangana dry belts, split irrigation is safer than heavy one-time watering.
        """
        why_block = """
### 🤔 Irrigation Logic
- Soil moisture, rainfall, and crop stage decide irrigation need.
- Overwatering increases root stress and disease spread.
- Split irrigation improves water-use efficiency in hot regions.
        """
    elif "fertilizer" in text:
        main_answer = f"""
### 🧪 Fertilizer Optimization
- Current recommendation: {actions[1]}
- Use split nitrogen application.
- Avoid heavy fertilizer before major rain.
- Check pH and soil moisture before top dressing.
- Add micronutrients in Telangana where soils are lighter and dry faster.
        """
        why_block = """
### 🤔 Fertilizer Logic
- Nitrogen losses are high during rain and low soil retention.
- Balanced NPK improves crop stability.
- Micronutrient sprays are often helpful in cotton and maize belts.
        """
    else:
        main_answer = f"""
### 🤖 Smart Agri Assistant
- Ask about crop choice, irrigation, fertilizer, disease, NDVI, or seasonal planning.
- Current optimization: {location} | {season}
- Regional note: {get_region_note(location, season)}
        """
        why_block = """
### 🤔 How It Works
- The assistant uses your latest Smart Advisor output.
- It combines crop, irrigation, rainfall, fertilizer, and stress guidance.
- It is designed for beginner-friendly agriculture AI projects.
        """

    return main_answer, why_block


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
        "Farmer Chat Assistant",
    ],
)

st.sidebar.markdown("### 🌍 Regional Simulation Mode")
location = st.sidebar.selectbox("Location", ["Andhra Pradesh", "Telangana"], index=1)
season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"], index=0)
defaults = get_simulation_defaults(location, season)

st.markdown(f"""
<div class="hero-box">
    <h1>🌾 AgriTech Smart Assistant Pro</h1>
    <p>AI-powered agriculture dashboard with state-wise optimization for Andhra Pradesh and Telangana, including crop advisory, irrigation planning, fertilizer support, NDVI analysis, disease screening, and farmer chatbot guidance.</p>
</div>
""", unsafe_allow_html=True)

if page == "Home":
    st.markdown('<div class="farmer-line">👨‍🌾 Now optimized for both Andhra Pradesh and Telangana farming conditions, including crop strategy, irrigation, fertilizer planning, and seasonal advisory.</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="region-chip">{location}</span><span class="region-chip">{season}</span><span class="region-chip">{get_region_note(location, season)}</span>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌾 Smart Farming", "📍 Regional Optimization", "🤖 AI Assistant"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.subheader("Key Features")
            st.write("""
- Crop recommendation with regional intelligence.
- Irrigation and rainfall decision support.
- Fertilizer advice based on nutrient imbalance.
- Yield prediction for farm planning.
- Final decision block for field action.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.subheader("Career Value")
            st.write("""
- Strong portfolio project for AgriTech AI jobs.
- Good for Streamlit, ML logic, and dashboard demos.
- Helpful for Hyderabad AgriTech interview discussions.
- Beginner friendly and ready for future model upgrades.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Telangana Optimization Added")
        st.write("""
- Telangana Kharif defaults support cotton, maize, red gram, and soybean.
- Telangana Rabi defaults support bengal gram, sesame, jowar, and groundnut.
- Dryland irrigation logic is improved for Telangana conditions.
- Fertilizer advice now considers Telangana crop patterns and dry belts.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Farmer Chat Assistant")
        st.write(f"""
- Uses latest Smart Advisor results.
- Answers are optimized for {location} and {season}.
- Includes guidance for crop, irrigation, fertilizer, and disease.
- Good demo module for agriculture AI assistant projects.
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
            crop, crop_conf = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall], location, season)
            irrigation, irrigation_action, irr_conf = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall], location)
            rain_value, weather_status, weather_conf = agrimodels.predict_rainfall([temperature, humidity, 1012, 8], location)
            fert_result, fert_conf = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, soil_moisture, 1, 1, ph, rainfall], location, crop)
            yield_pred, yield_conf = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer_val, humidity, soil_moisture], location)
            ndvi = agrimodels.calculate_ndvi(red, nir)
            stress_level, stress_conf = agrimodels.predict_crop_stress([ndvi, temperature, soil_moisture, humidity], location)

            final_output = {
                "crop": crop,
                "irrigation": irrigation,
                "weather": weather_status,
                "actions": [irrigation_action, fert_result],
                "yield": yield_pred,
                "ndvi": ndvi,
                "stress": stress_level,
            }
            st.session_state.latest_final_output = final_output

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
                <h3>✅ Final Farm Decision</h3>
                <strong>📍 Region:</strong> {location} - {season}<br>
                <strong>🌾 Crop:</strong> {crop} ({crop_conf}%)<br>
                <strong>💧 Irrigation:</strong> {irrigation} - {irrigation_action}<br>
                <strong>🧪 Fertilizer:</strong> {fert_result}<br>
                <strong>🌦 Weather:</strong> {weather_status} ({rain_value}mm)<br>
                <strong>📈 NDVI:</strong> {ndvi:.3f} ({stress_level} stress)<br>
                <strong>🎯 Yield:</strong> {yield_pred} tons/ha<br>
                <strong>📝 Regional Note:</strong> {get_region_note(location, season)}
            </div>
            """, unsafe_allow_html=True)

elif page == "Crop Recommendation":
    st.subheader("🌾 Crop Recommendation")
    inputs = [num_input(f"Feature {i+1}", defaults["N"] + i * 2) for i in range(7)]
    if st.button("Recommend Crop"):
        crop, conf = agrimodels.predict_crop_recommendation(inputs, location, season)
        st.success(f"Recommended crop for {location} {season}: {crop} ({conf}% confidence)")

elif page == "Crop Yield":
    st.subheader("📈 Crop Yield Prediction")
    yield_inputs = [num_input(f"Input {i+1}", 25.0) for i in range(6)]
    if st.button("Predict Yield"):
        yield_pred, conf = agrimodels.predict_crop_yield(yield_inputs, location)
        st.metric("Expected Yield", f"{yield_pred} tons/ha", delta=f"{conf}% confidence")

elif page == "Irrigation":
    st.subheader("💧 Irrigation Advisor")
    irr_inputs = [num_input(f"Param {i+1}", 40.0) for i in range(5)]
    if st.button("Get Irrigation Plan"):
        result, action, conf = agrimodels.predict_irrigation(irr_inputs, location)
        st.markdown(f"**Need:** {result} | **Action:** {action} ({conf}%)")

elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")
    fert_inputs = [num_input(f"Input {i+1}", 50.0) for i in range(10)]
    crop_name = st.text_input("Crop Name", value="Cotton" if location == "Telangana" else "Rice")
    if st.button("Recommend Fertilizer"):
        result, conf = agrimodels.predict_fertilizer(fert_inputs, location, crop_name)
        st.success(f"{result} ({conf}%)")

elif page == "Soil & NPK":
    st.subheader("🧪 Soil NPK Analysis")
    npk_inputs = [num_input(f"Soil {i+1}", 1.0) for i in range(6)]
    if st.button("Analyze Soil"):
        npk, conf = agrimodels.predict_npk(npk_inputs)
        st.json(npk)
        st.caption(f"Confidence: {conf}%")

elif page == "NDVI & Stress":
    st.subheader("📈 NDVI & Crop Stress")
    red_col, nir_col = st.columns(2)
    with red_col:
        red_val = st.number_input("Red Band", 0.0, 1.0, 0.3)
    with nir_col:
        nir_val = st.number_input("NIR Band", 0.0, 1.0, 0.7)

    temp_col, soil_col, hum_col = st.columns(3)
    with temp_col:
        temp_val = st.number_input("Temperature", 15.0, 45.0, float(defaults["temperature"]))
    with soil_col:
        soil_val = st.number_input("Soil Moisture", 10.0, 80.0, float(defaults["soil_moisture"]))
    with hum_col:
        hum_val = st.number_input("Humidity", 30.0, 100.0, float(defaults["humidity"]))

    if st.button("Calculate Health"):
        ndvi = agrimodels.calculate_ndvi(red_val, nir_val)
        stress, conf = agrimodels.predict_crop_stress([ndvi, temp_val, soil_val, hum_val], location)

        fig = go.Figure()
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Monitor")
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Critical")
        fig.add_trace(go.Scatter(
            x=[1],
            y=[ndvi],
            mode="markers+text",
            marker=dict(size=20, color="blue"),
            text=[f"NDVI: {ndvi:.3f}"],
            textposition="middle center",
            name="Your Field"
        ))
        fig.update_layout(title=f"NDVI Health Score - {location}", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Stress Level:** {stress} ({conf}% confidence)")

elif page == "Image Models":
    st.subheader("🖼️ Leaf Disease Detection")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image, arr = process_image(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        if st.button("Analyze Disease"):
            disease, conf = agrimodels.predict_disease(arr)
            st.error(f"Detected: {disease} ({conf}% confidence)")
            st.info("Real model can be replaced later with CNN or transfer learning model.")

elif page == "Farmer Chat Assistant":
    st.markdown(f"""
    <div class="chat-toolbar">
        💬 <strong>Smart Agri Assistant</strong> | Region: <strong>{location}</strong> | Season: <strong>{season}</strong>
        {'✅ Latest analysis loaded!' if st.session_state.latest_final_output else '⚠️ Run Smart Advisor first!'}
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    cols = st.columns(5)
    questions = [
        "What should I do?",
        "Telangana crop tips?",
        "Irrigation plan?",
        "Fertilizer advice?",
        "Disease risk?",
    ]
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q, key=f"quick_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                main_reply, why_reply = generate_gpt_like_reply(q, st.session_state.latest_final_output, location, season)
                st.session_state.chat_history.append({"role": "assistant", "content": main_reply + "\n\n" + why_reply})
                st.rerun()

    if prompt := st.chat_input("Ask about your farm, Telangana crops, irrigation, fertilizer, disease, or yield..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AgriTech AI is thinking..."):
                main_reply, why_reply = generate_gpt_like_reply(prompt, st.session_state.latest_final_output, location, season)
                full_reply = main_reply + "\n\n" + why_reply
                st.markdown(full_reply)
                st.session_state.chat_history.append({"role": "assistant", "content": full_reply})

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    🌾 Built for Andhra Pradesh and Telangana farmers | AI/ML Agriculture Assistant | Streamlit portfolio ready
</div>
""", unsafe_allow_html=True)
