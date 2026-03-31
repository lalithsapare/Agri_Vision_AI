import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AgriTech Smart Assistant",
    page_icon="🌾",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.main {
    background: #f6fff8;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14532d 0%, #1f7a46 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    border: 1px solid #d9efe0;
    margin-bottom: 15px;
}
.big-title {
    background: linear-gradient(90deg, #166534, #22a861);
    color: white;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ENGINE ----------------
class AgritechModels:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        os.makedirs(models_dir, exist_ok=True)
        self.load_all_models()

    def load_or_dummy(self, key, model_type="regressor"):
        path = os.path.join(self.models_dir, f"{key}.pkl")
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                pass

        if model_type == "classifier":
            return None
        return None

    def load_all_models(self):
        self.models["crop_recommendation"] = self.load_or_dummy("crop_recommendation", "classifier")
        self.models["crop_yield"] = self.load_or_dummy("crop_yield", "regressor")
        self.models["irrigation"] = self.load_or_dummy("irrigation", "classifier")
        self.models["fertilizer"] = self.load_or_dummy("fertilizer", "classifier")
        self.models["temperature"] = self.load_or_dummy("temperature", "regressor")
        self.models["rainfall"] = self.load_or_dummy("rainfall", "regressor")
        self.models["humidity"] = self.load_or_dummy("humidity", "regressor")
        self.models["soil_ph"] = self.load_or_dummy("soil_ph", "regressor")
        self.models["price_prediction"] = self.load_or_dummy("price_prediction", "regressor")
        self.models["harvest_time"] = self.load_or_dummy("harvest_time", "regressor")
        self.models["ndvi"] = self.load_or_dummy("ndvi", "regressor")
        self.models["crop_stress"] = self.load_or_dummy("crop_stress", "classifier")
        self.models["npk_nitrogen"] = self.load_or_dummy("npk_nitrogen", "regressor")
        self.models["npk_phosphorus"] = self.load_or_dummy("npk_phosphorus", "regressor")
        self.models["npk_potassium"] = self.load_or_dummy("npk_potassium", "regressor")

    def predict_crop_recommendation(self, features):
        crops = [
            "Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans",
            "Mungbean", "Blackgram", "Lentil", "Pomegranate", "Banana", "Mango",
            "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya",
            "Coconut", "Cotton", "Jute", "Coffee"
        ]
        score = int(sum(features)) % len(crops)
        return crops[score]

    def predict_crop_yield(self, features):
        ph, temp, rainfall, fertilizer, humidity, soil_moisture = features
        pred = (temp * 0.8) + (rainfall * 0.03) + (fertilizer * 0.05) + (humidity * 0.1) + (soil_moisture * 0.2) - abs(ph - 6.5) * 2
        return round(max(pred, 0), 2)

    def predict_irrigation(self, features):
        soil_moisture, temp, humidity, ph, rainfall = features
        if soil_moisture < 35 and rainfall < 120:
            return "🚿 Irrigate Immediately"
        return "✅ No Irrigation Needed"

    def predict_fertilizer(self, features):
        N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall = features
        if N < 50:
            return "Urea"
        elif P < 40:
            return "DAP"
        return "MOP"

    def predict_temperature(self, features):
        return round(np.mean(features), 2)

    def predict_rainfall(self, features):
        temp, humidity, pressure, wind_speed = features
        pred = (humidity * 0.7) - (temp * 0.2) + (wind_speed * 0.5) + ((1015 - pressure) * 1.2)
        return round(max(pred, 0), 2)

    def predict_humidity(self, features):
        temp, pressure, wind_speed = features
        pred = 85 - (temp * 0.7) + ((1015 - pressure) * 0.5) + (wind_speed * 0.2)
        return round(max(min(pred, 100), 0), 2)

    def predict_ph(self, features):
        soil_moisture, organic_matter, temp, rainfall = features
        pred = 6.5 + (organic_matter * 0.2) - (rainfall * 0.001) + (soil_moisture * 0.01)
        return round(pred, 2)

    def predict_price(self, features):
        year, month, market_code, arrival_qty, demand_index, crop_code = features
        pred = 1200 + (demand_index * 500) - (arrival_qty * 0.8) + (month * 10)
        return round(max(pred, 100), 2)

    def predict_harvest_time(self, features):
        days_after_sowing, temp, humidity, ph, rainfall, soil_moisture = features
        pred = days_after_sowing + (100 - soil_moisture) * 0.2 - (temp * 0.1)
        return round(max(pred, 1), 1)

    def predict_npk(self, features):
        ph, ec, organic_carbon, moisture, temp, rainfall = features
        n = round((organic_carbon * 40) + (moisture * 0.8), 2)
        p = round((ec * 20) + (ph * 3), 2)
        k = round((rainfall * 0.1) + (temp * 0.5), 2)
        return {"Nitrogen": n, "Phosphorus": p, "Potassium": k}

    def predict_ndvi(self, features):
        red, nir = features
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_crop_stress(self, features):
        ndvi, temp, soil_moisture, humidity = features
        if ndvi < 0.4 or soil_moisture < 30 or temp > 35:
            return "🚨 High Stress"
        return "✅ Healthy"

    def calculate_ndvi(self, red, nir):
        return (nir - red) / (nir + red + 1e-8)

    def predict_disease(self, image_array):
        labels = ["Healthy", "Leaf Spot", "Rust", "Blight", "Powdery Mildew"]
        return np.random.choice(labels)

    def predict_leaf(self, image_array):
        labels = ["Tomato", "Potato", "Cotton", "Maize", "Rice"]
        return np.random.choice(labels)

    def predict_weed(self, image_array):
        labels = ["Weed Detected", "No Weed"]
        return np.random.choice(labels)

    def predict_soil(self, image_array):
        labels = ["Black Soil", "Alluvial Soil", "Red Soil", "Sandy Soil"]
        return np.random.choice(labels)

    def predict_nutrient(self, image_array):
        labels = ["Nitrogen Deficiency", "Potassium Deficiency", "Phosphorus Deficiency", "Healthy"]
        return np.random.choice(labels)

@st.cache_resource
def load_agritech_models():
    return AgritechModels("models")

agrimodels = load_agritech_models()

# ---------------- HELPERS ----------------
def num_input(label, value=0.0):
    return st.number_input(label, value=float(value), step=0.1)

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr

def chatbot_reply(prompt):
    p = prompt.lower()
    if "crop" in p:
        return "Use Crop Recommendation and enter N, P, K, temperature, humidity, pH, and rainfall."
    if "yield" in p:
        return "Use Crop Yield and enter pH, temperature, rainfall, fertilizer, humidity, and soil moisture."
    if "disease" in p:
        return "Open Image Models and upload a leaf image for disease prediction."
    if "ndvi" in p:
        return "Use NDVI & Stress to calculate NDVI from Red and NIR values."
    if "irrigation" in p:
        return "Use Irrigation Prediction with soil moisture, temperature, humidity, pH, and rainfall."
    return "I can help with crop recommendation, yield, irrigation, fertilizer, NDVI, disease, soil, and stress prediction."

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 AgriTech Menu")
page = st.sidebar.radio(
    "Choose Module",
    [
        "Home",
        "Chatbot",
        "Crop Recommendation",
        "Crop Yield",
        "Irrigation",
        "Fertilizer",
        "Weather",
        "Soil pH",
        "Price Prediction",
        "Harvest Time",
        "NPK Prediction",
        "NDVI & Stress",
        "Image Models"
    ]
)

# ---------------- HEADER ----------------
st.markdown("""
<div class="big-title">
    <h1>🌾 AgriTech Smart Assistant</h1>
    <p>All-in-one agriculture AI app for crop recommendation, yield, irrigation, fertilizer, NDVI, disease, soil, weather, and price analysis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PAGES ----------------
if page == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Platform Features")
        st.write("""
- Crop recommendation.
- Yield prediction.
- Irrigation support.
- Fertilizer suggestion.
- NDVI calculation.
- Disease and image-based analysis.
- Soil and pH estimation.
- Crop stress prediction.
- Market price estimation.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Use Cases")
        st.write("""
- Student final-year project.
- Agritech portfolio.
- Farmer advisory demo.
- AI/ML agriculture showcase.
- Streamlit cloud deployment project.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Chatbot":
    st.subheader("💬 Agri Chatbot")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    prompt = st.chat_input("Ask about crop, irrigation, NDVI, disease, fertilizer...")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)
        reply = chatbot_reply(prompt)
        st.session_state.chat_history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.write(reply)

elif page == "Crop Recommendation":
    st.subheader("🌱 Crop Recommendation")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = num_input("Nitrogen", 90)
        P = num_input("Phosphorus", 42)
        K = num_input("Potassium", 43)
    with c2:
        temperature = num_input("Temperature", 25)
        humidity = num_input("Humidity", 80)
    with c3:
        ph = num_input("pH", 6.5)
        rainfall = num_input("Rainfall", 200)
    if st.button("Predict Crop"):
        result = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall])
        st.success(f"Recommended Crop: {result}")

elif page == "Crop Yield":
    st.subheader("🌾 Crop Yield")
    c1, c2, c3 = st.columns(3)
    with c1:
        ph = num_input("pH", 6.5)
        temperature = num_input("Temperature", 26)
    with c2:
        rainfall = num_input("Rainfall", 180)
        fertilizer = num_input("Fertilizer", 120)
    with c3:
        humidity = num_input("Humidity", 75)
        soil_moisture = num_input("Soil Moisture", 40)
    if st.button("Predict Yield"):
        result = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer, humidity, soil_moisture])
        st.success(f"Predicted Yield: {result}")

elif page == "Irrigation":
    st.subheader("💧 Irrigation Prediction")
    c1, c2 = st.columns(2)
    with c1:
        soil_moisture = num_input("Soil Moisture", 35)
        temperature = num_input("Temperature", 28)
        humidity = num_input("Humidity", 70)
    with c2:
        ph = num_input("pH", 6.8)
        rainfall = num_input("Rainfall", 120)
    if st.button("Predict Irrigation"):
        result = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall])
        st.success(result)

elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = num_input("N", 80)
        P = num_input("P", 40)
        K = num_input("K", 40)
        temperature = num_input("Temperature", 27)
    with c2:
        humidity = num_input("Humidity", 70)
        moisture = num_input("Moisture", 30)
        soil_type = num_input("Soil Type Code", 1)
    with c3:
        crop_type = num_input("Crop Type Code", 1)
        ph = num_input("pH", 6.5)
        rainfall = num_input("Rainfall", 150)
    if st.button("Predict Fertilizer"):
        result = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall])
        st.success(f"Recommended Fertilizer: {result}")

elif page == "Weather":
    st.subheader("⛅ Weather Analysis")
    tab1, tab2, tab3 = st.tabs(["Temperature", "Rainfall", "Humidity"])
    with tab1:
        s1 = num_input("Sensor 1", 25)
        s2 = num_input("Sensor 2", 27)
        if st.button("Predict Temperature"):
            st.success(f"Predicted Temperature: {agrimodels.predict_temperature([s1, s2])}")
    with tab2:
        t = num_input("Temp", 29)
        h = num_input("Humidity ", 75)
        p = num_input("Pressure", 1012)
        w = num_input("Wind Speed", 10)
        if st.button("Predict Rainfall"):
            st.success(f"Predicted Rainfall: {agrimodels.predict_rainfall([t, h, p, w])}")
    with tab3:
        t2 = num_input("Temp ", 29)
        p2 = num_input("Pressure ", 1011)
        w2 = num_input("Wind Speed ", 8)
        if st.button("Predict Humidity"):
            st.success(f"Predicted Humidity: {agrimodels.predict_humidity([t2, p2, w2])}")

elif page == "Soil pH":
    st.subheader("🌍 Soil pH Prediction")
    soil_moisture = num_input("Soil Moisture", 40)
    organic_matter = num_input("Organic Matter", 2.2)
    temperature = num_input("Temperature", 27)
    rainfall = num_input("Rainfall", 140)
    if st.button("Predict Soil pH"):
        st.success(f"Predicted pH: {agrimodels.predict_ph([soil_moisture, organic_matter, temperature, rainfall])}")

elif page == "Price Prediction":
    st.subheader("💰 Price Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        year = num_input("Year", 2026)
        month = num_input("Month", 4)
    with c2:
        market_code = num_input("Market Code", 101)
        arrival_qty = num_input("Arrival Quantity", 200)
    with c3:
        demand_index = num_input("Demand Index", 1.2)
        crop_code = num_input("Crop Code", 10)
    if st.button("Predict Price"):
        st.success(f"Predicted Price: ₹{agrimodels.predict_price([year, month, market_code, arrival_qty, demand_index, crop_code])}")

elif page == "Harvest Time":
    st.subheader("🕒 Harvest Time Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        days_after_sowing = num_input("Days After Sowing", 90)
        temperature = num_input("Temperature", 28)
    with c2:
        humidity = num_input("Humidity", 76)
        ph = num_input("pH", 6.4)
    with c3:
        rainfall = num_input("Rainfall", 130)
        soil_moisture = num_input("Soil Moisture", 35)
    if st.button("Predict Harvest Time"):
        st.success(f"Harvest in: {agrimodels.predict_harvest_time([days_after_sowing, temperature, humidity, ph, rainfall, soil_moisture])} days")

elif page == "NPK Prediction":
    st.subheader("🧬 NPK Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        ph = num_input("pH", 6.5)
        ec = num_input("EC", 1.2)
    with c2:
        organic_carbon = num_input("Organic Carbon", 0.8)
        moisture = num_input("Moisture", 30)
    with c3:
        temperature = num_input("Temperature", 27)
        rainfall = num_input("Rainfall", 140)
    if st.button("Predict NPK"):
        result = agrimodels.predict_npk([ph, ec, organic_carbon, moisture, temperature, rainfall])
        st.json(result)

elif page == "NDVI & Stress":
    st.subheader("📈 NDVI and Stress")
    tab1, tab2 = st.tabs(["NDVI", "Stress"])
    with tab1:
        red = num_input("Red Band", 0.3)
        nir = num_input("NIR Band", 0.7)
        if st.button("Calculate NDVI"):
            ndvi = agrimodels.calculate_ndvi(red, nir)
            st.success(f"NDVI Value: {ndvi:.4f}")
            chart_df = pd.DataFrame({
                "Band": ["Red", "NIR"],
                "Value": [red, nir]
            })
            fig = px.bar(chart_df, x="Band", y="Value", color="Band", title="Band Values")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ndvi_val = num_input("NDVI Value", 0.65)
        temperature = num_input("Temperature ", 28)
        soil_moisture = num_input("Soil Moisture ", 32)
        humidity = num_input("Humidity  ", 72)
        if st.button("Predict Stress"):
            st.success(agrimodels.predict_crop_stress([ndvi_val, temperature, soil_moisture, humidity]))

elif page == "Image Models":
    st.subheader("🖼️ Image-Based Models")
    model_choice = st.selectbox(
        "Choose Image Model",
        ["Plant Disease", "Leaf Classification", "Weed Detection", "Soil Classification", "Nutrient Deficiency"]
    )
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image, image_array = process_image(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Predict Image"):
            if model_choice == "Plant Disease":
                result = agrimodels.predict_disease(image_array)
            elif model_choice == "Leaf Classification":
                result = agrimodels.predict_leaf(image_array)
            elif model_choice == "Weed Detection":
                result = agrimodels.predict_weed(image_array)
            elif model_choice == "Soil Classification":
                result = agrimodels.predict_soil(image_array)
            else:
                result = agrimodels.predict_nutrient(image_array)
            st.success(f"Prediction: {result}")
