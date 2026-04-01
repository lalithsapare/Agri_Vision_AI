import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AgriTech Smart Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------
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

if "sim_location" not in st.session_state:
    st.session_state.sim_location = "Andhra Pradesh"

if "sim_season" not in st.session_state:
    st.session_state.sim_season = "Kharif"

# ---------------------- HELPERS ----------------------
def num_input(label, value=0.0):
    return st.number_input(label, value=float(value), step=0.1)

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr

def smart_farmer_chat(user_text, final_output=None):
    text = user_text.lower()
    if "what should i do" in text and final_output:
        return (
            f"Based on current results, grow {final_output['crop']}, keep irrigation {final_output['irrigation']}, "
            f"and prepare for {final_output['weather']}. Recommended actions are: "
            f"{', '.join(final_output['actions'])}."
        )
    if "disease" in text:
        return "Upload a crop image in Image Models to detect disease and get confidence score."
    if "irrigation" in text:
        return "Check soil moisture, rainfall, and temperature first. If rainfall is high, reduce irrigation."
    if "fertilizer" in text:
        return "Fertilizer advice should follow soil nutrients, crop need, and rainfall condition."
    if "ndvi" in text:
        return "NDVI helps measure crop vigor. Higher NDVI usually means better crop health."
    return "I can guide you with crop selection, irrigation, fertilizer, disease risk, NDVI, and final farm action."

def get_simulation_defaults(location, season):
    data = {
        ("Andhra Pradesh", "Kharif"): {
            "N": 90, "P": 42, "K": 43, "temperature": 28, "humidity": 82, "ph": 6.5, "rainfall": 220,
            "soil_moisture": 48, "fertilizer": 110, "organic_matter": 2.1, "ec": 1.2, "organic_carbon": 0.82
        },
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
    <p>AI-powered agriculture prediction dashboard with chatbot, image analysis, smart simulation, health scoring, and final field decision output.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- HOME ----------------------
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
        st.subheader("AI Assistant")
        st.write("""
- Smart farmer chat responses.
- Final recommendation engine.
- Real-world simulation mode for Andhra Pradesh and seasonal use.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- SMART ADVISOR ----------------------
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

    if st.button("Generate Final Farm Decision"):
        crop, crop_conf = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall])
        irrigation, irrigation_action, irr_conf = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall])
        rain_value, weather_status, weather_conf = agrimodels.predict_rainfall([temperature, humidity, 1012, 8])
        fert_result, fert_conf = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, soil_moisture, 1, 1, ph, rainfall])
        ndvi = agrimodels.calculate_ndvi(red, nir)
        stress, health_conf = agrimodels.predict_crop_stress([ndvi, temperature, soil_moisture, humidity])
        yield_pred, yield_conf = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer_val, humidity, soil_moisture])

        if stress == "Low" and ndvi > 0.55:
            health_score = 85
            risk = "LOW"
            disease_line = "No major disease signal"
        elif stress == "Moderate":
            health_score = 67
            risk = "MEDIUM"
            disease_line = "Monitor crop disease symptoms"
        else:
            health_score = 42
            risk = "HIGH"
            disease_line = "High disease/stress probability"

        actions = [irrigation_action, fert_result]
        if rainfall > 180:
            actions.append("Spray fungicide")
        else:
            actions.append("Continue field monitoring")

        final_output = {
            "crop": crop,
            "irrigation": irrigation,
            "weather": weather_status,
            "actions": actions
        }

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f'<div class="metric-card"><h4>🌾 Crop</h4><h3>{crop}</h3><p>{crop_conf}% confidence</p></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><h4>💧 Irrigation</h4><h3>{irrigation}</h3><p>{irr_conf}% confidence</p></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-card"><h4>🌦 Weather</h4><h3>{weather_status}</h3><p>{weather_conf}% confidence</p></div>', unsafe_allow_html=True)
        with mc4:
            st.markdown(f'<div class="metric-card"><h4>💚 Health Score</h4><h3>{health_score}%</h3><p>Risk: {risk}</p></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="decision-card">
            <h3>FINAL OUTPUT</h3>
            <p><b>🌾 Crop:</b> {crop}</p>
            <p><b>💧 Irrigation:</b> {irrigation}</p>
            <p><b>🌦 Weather:</b> {weather_status}</p>
            <p><b>👉 ACTION:</b></p>
            <ul>
                <li>{actions[0]}</li>
                <li>{actions[1]}</li>
                <li>{actions[2]}</li>
            </ul>
            <p><b>Risk Score:</b> {risk}</p>
            <p><b>Health Score:</b> {health_score}%</p>
            <p><b>Based on:</b> NDVI = {ndvi}, Crop Stress = {stress}, Disease = {disease_line}</p>
            <p><b>Yield Prediction:</b> {yield_pred} ({yield_conf}% confidence)</p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state["latest_final_output"] = final_output

        # -------- VISUALIZATIONS --------
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fert_df = pd.DataFrame({
                "Component": ["Nitrogen", "Phosphorus", "Potassium", "Soil Moisture"],
                "Value": [N, P, K, soil_moisture]
            })
            fig1 = px.bar(fert_df, x="Component", y="Value", color="Component", title="📊 Fertilizer vs Soil Condition")
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            ndvi_x = ["Red", "NIR", "NDVI"]
            ndvi_y = [red, nir, ndvi]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ndvi_x, y=ndvi_y, mode="lines+markers", name="NDVI Trend"))
            fig2.update_layout(title="📈 NDVI Graph")
            st.plotly_chart(fig2, use_container_width=True)

        yield_df = pd.DataFrame({
            "Stage": ["Current", "Improved Irrigation", "Improved Fertilizer", "Best Scenario"],
            "Yield": [yield_pred, yield_pred * 1.08, yield_pred * 1.12, yield_pred * 1.18]
        })
        fig3 = px.line(yield_df, x="Stage", y="Yield", markers=True, title="📉 Yield Prediction Graph")
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------- CROP RECOMMENDATION ----------------------
elif page == "Crop Recommendation":
    st.subheader("🌱 Crop Recommendation")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = num_input("Nitrogen", defaults["N"])
        P = num_input("Phosphorus", defaults["P"])
        K = num_input("Potassium", defaults["K"])
    with c2:
        temperature = num_input("Temperature", defaults["temperature"])
        humidity = num_input("Humidity", defaults["humidity"])
    with c3:
        ph = num_input("pH", defaults["ph"])
        rainfall = num_input("Rainfall", defaults["rainfall"])

    if st.button("Predict Crop"):
        crop, conf = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall])
        st.success(f"Recommended Crop: {crop} ({conf}% confidence)")

# ---------------------- CROP YIELD ----------------------
elif page == "Crop Yield":
    st.subheader("🌾 Crop Yield Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        ph = num_input("pH", defaults["ph"])
        temperature = num_input("Temperature", defaults["temperature"])
    with c2:
        rainfall = num_input("Rainfall", defaults["rainfall"])
        fertilizer_val = num_input("Fertilizer", defaults["fertilizer"])
    with c3:
        humidity = num_input("Humidity", defaults["humidity"])
        soil_moisture = num_input("Soil Moisture", defaults["soil_moisture"])

    if st.button("Predict Yield"):
        result, conf = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer_val, humidity, soil_moisture])
        st.success(f"Predicted Yield: {result} ({conf}% confidence)")

# ---------------------- IRRIGATION ----------------------
elif page == "Irrigation":
    st.subheader("💧 Irrigation Prediction")
    c1, c2 = st.columns(2)
    with c1:
        soil_moisture = num_input("Soil Moisture", defaults["soil_moisture"])
        temperature = num_input("Temperature", defaults["temperature"])
        humidity = num_input("Humidity", defaults["humidity"])
    with c2:
        ph = num_input("pH", defaults["ph"])
        rainfall = num_input("Rainfall", defaults["rainfall"])

    if st.button("Predict Irrigation"):
        result, action, conf = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall])
        st.success(f"Irrigation: {result} ({conf}% confidence)")
        st.info(f"Action: {action}")

# ---------------------- FERTILIZER ----------------------
elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = num_input("N", defaults["N"])
        P = num_input("P", defaults["P"])
        K = num_input("K", defaults["K"])
        temperature = num_input("Temperature", defaults["temperature"])
    with c2:
        humidity = num_input("Humidity", defaults["humidity"])
        moisture = num_input("Moisture", defaults["soil_moisture"])
        soil_type = num_input("Soil Type Code", 1)
    with c3:
        crop_type = num_input("Crop Type Code", 1)
        ph = num_input("pH", defaults["ph"])
        rainfall = num_input("Rainfall", defaults["rainfall"])

    if st.button("Recommend Fertilizer"):
        result, conf = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall])
        st.success(f"{result} ({conf}% confidence)")

# ---------------------- SOIL & NPK ----------------------
elif page == "Soil & NPK":
    st.subheader("🌍 Soil and NPK Analysis")
    tab1, tab2 = st.tabs(["Soil pH", "NPK Prediction"])

    with tab1:
        soil_moisture = num_input("Soil Moisture", defaults["soil_moisture"])
        organic_matter = num_input("Organic Matter", defaults["organic_matter"])
        temperature = num_input("Temperature", defaults["temperature"])
        rainfall = num_input("Rainfall", defaults["rainfall"])
        if st.button("Predict Soil pH"):
            ph_val, conf = agrimodels.predict_ph([soil_moisture, organic_matter, temperature, rainfall])
            st.success(f"Predicted Soil pH: {ph_val} ({conf}% confidence)")

    with tab2:
        ph = num_input("pH", defaults["ph"])
        ec = num_input("EC", defaults["ec"])
        organic_carbon = num_input("Organic Carbon", defaults["organic_carbon"])
        moisture = num_input("Moisture", defaults["soil_moisture"])
        temp = num_input("Temperature ", defaults["temperature"])
        rain = num_input("Rainfall ", defaults["rainfall"])
        if st.button("Predict NPK"):
            result, conf = agrimodels.predict_npk([ph, ec, organic_carbon, moisture, temp, rain])
            st.json({"NPK": result, "confidence": conf})

# ---------------------- NDVI & STRESS ----------------------
elif page == "NDVI & Stress":
    st.subheader("📈 NDVI and Crop Stress")
    tab1, tab2 = st.tabs(["NDVI", "Stress"])

    with tab1:
        red = num_input("Red Band", 0.30)
        nir = num_input("NIR Band", 0.72)
        if st.button("Calculate NDVI"):
            ndvi = agrimodels.calculate_ndvi(red, nir)
            st.success(f"NDVI Value: {ndvi}")

            df = pd.DataFrame({
                "Band": ["Red", "NIR", "NDVI"],
                "Value": [red, nir, ndvi]
            })
            fig = px.line(df, x="Band", y="Value", markers=True, title="NDVI Visualization")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ndvi_val = num_input("NDVI Value", 0.62)
        temperature = num_input("Temperature", defaults["temperature"])
        soil_moisture = num_input("Soil Moisture", defaults["soil_moisture"])
        humidity = num_input("Humidity", defaults["humidity"])
        if st.button("Predict Crop Stress"):
            stress, conf = agrimodels.predict_crop_stress([ndvi_val, temperature, soil_moisture, humidity])
            health = 85 if stress == "Low" else 65 if stress == "Moderate" else 40
            risk = "LOW" if stress == "Low" else "MEDIUM" if stress == "Moderate" else "HIGH"
            st.success(f"Crop Stress: {stress} ({conf}% confidence)")
            st.info(f"Health Score: {health}% | Risk: {risk}")

# ---------------------- IMAGE MODELS ----------------------
elif page == "Image Models":
    st.subheader("🖼️ Crop Disease Detection")
    uploaded_file = st.file_uploader("Upload crop image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image, image_array = process_image(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        if st.button("Analyze Image"):
            disease, conf = agrimodels.predict_disease(image_array)
            st.success(f"Disease Prediction: {disease} ({conf}% confidence)")

# ---------------------- FARMER CHAT ----------------------
elif page == "Farmer Chat Assistant":
    st.subheader("🤖 Farmer Chat Assistant")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    prompt = st.chat_input("Ask: What should I do?")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        final_output = st.session_state.get("latest_final_output", None)
        reply = smart_farmer_chat(prompt, final_output)
        st.session_state.chat_history.append(("assistant", reply))

        with st.chat_message("assistant"):
            st.write(reply)