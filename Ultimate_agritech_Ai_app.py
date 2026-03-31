import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="AgriTech Smart Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f4fff6 0%, #eefbf3 50%, #f8fffc 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3d2e 0%, #145a41 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.agri-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border: 1px solid #dff3e7;
    margin-bottom: 16px;
}
.title-box {
    background: linear-gradient(90deg, #1b7f5b, #35a37b);
    padding: 18px;
    border-radius: 16px;
    color: white;
    box-shadow: 0 8px 18px rgba(53,163,123,0.25);
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)


class AgritechModels:
    def __init__(self, models_dir="Agritech_models"):
        self.models_dir = models_dir
        self.models = {}
        self.model_status = {}
        self.model_files = {
            "crop_recommendation": "crop_recommendation.pkl",
            "crop_yield": "crop_yield.pkl",
            "irrigation": "irrigation.pkl",
            "fertilizer": "fertilizer.pkl",
            "soil_ph": "soil_ph.pkl",
            "price_prediction": "price_prediction.pkl",
            "harvest_time": "harvest_time.pkl",
            "npk": "npk.pkl"
        }
        self.load_models()

    def load_models(self):
        for model_key, filename in self.model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            try:
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    self.model_status[model_key] = f"Loaded: {filename}"
                else:
                    self.models[model_key] = None
                    self.model_status[model_key] = f"Missing: {filename}"
            except Exception as e:
                self.models[model_key] = None
                self.model_status[model_key] = f"Error loading {filename}: {str(e)}"

    def get_status(self):
        return self.model_status

    def _predict(self, model_key, features, label="Prediction"):
        model = self.models.get(model_key)
        if model is None:
            return f"{label} model unavailable"

        try:
            features = np.array(features, dtype=float).reshape(1, -1)
            pred = model.predict(features)[0]

            if isinstance(pred, (np.floating, float)):
                return round(float(pred), 4)
            if isinstance(pred, (np.integer, int)):
                return int(pred)
            return str(pred)
        except Exception as e:
            return f"{label} error: {str(e)}"

    def predict_crop_recommendation(self, features):
        return self._predict("crop_recommendation", features, "Crop recommendation")

    def predict_crop_yield(self, features):
        return self._predict("crop_yield", features, "Crop yield")

    def predict_irrigation(self, features):
        return self._predict("irrigation", features, "Irrigation")

    def predict_fertilizer(self, features):
        return self._predict("fertilizer", features, "Fertilizer")

    def predict_ph(self, features):
        return self._predict("soil_ph", features, "Soil pH")

    def predict_price(self, features):
        return self._predict("price_prediction", features, "Price")

    def predict_harvest_time(self, features):
        return self._predict("harvest_time", features, "Harvest time")

    def predict_npk(self, features):
        return self._predict("npk", features, "NPK")

    def calculate_ndvi(self, red, nir):
        try:
            red = float(red)
            nir = float(nir)
            denom = nir + red
            if denom == 0:
                return 0.0
            return (nir - red) / (nir + red)
        except Exception:
            return 0.0

    def predict_ndvi(self, features):
        try:
            red = float(features[0])
            nir = float(features[1])
            return round(self.calculate_ndvi(red, nir), 4)
        except Exception as e:
            return f"NDVI error: {str(e)}"

    def predict_crop_stress(self, features):
        try:
            ndvi = float(features[0])
            temperature = float(features[1])
            soil_moisture = float(features[2])
            humidity = float(features[3])

            if ndvi < 0.2 or soil_moisture < 20:
                return "High Stress"
            elif ndvi < 0.4 or temperature > 35 or humidity < 35:
                return "Moderate Stress"
            else:
                return "Low Stress"
        except Exception as e:
            return f"Stress prediction error: {str(e)}"


@st.cache_resource
def load_agritech_models():
    return AgritechModels("Agritech_models")


try:
    agrimodels = load_agritech_models()
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def show_model_card(model_name, desc):
    with st.expander(f"🔧 {model_name}", expanded=False):
        st.caption(desc)

def safe_float_input(label, value=0.0):
    return st.number_input(label, value=float(value), step=0.1, format="%.2f")

def preprocess_uploaded_image(uploaded_file, size=(224, 224)):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(size)
    image_array = np.array(image, dtype=np.float32)
    return image, image_array

def bot_response_from_prompt(prompt):
    text = prompt.lower()

    if "crop" in text and "recommend" in text:
        return "Use Crop Recommendation and enter N, P, K, temperature, humidity, pH, and rainfall."
    elif "yield" in text:
        return "Use Crop Yield Prediction with pH, temperature, rainfall, fertilizer, humidity, and soil moisture."
    elif "irrigation" in text:
        return "Use Irrigation Prediction with soil moisture, temperature, humidity, pH, and rainfall."
    elif "fertilizer" in text:
        return "Use Fertilizer Recommendation and enter nutrient plus soil details."
    elif "ndvi" in text:
        return "Use NDVI & Stress with Red and NIR values."
    elif "stress" in text:
        return "Use Stress Prediction with NDVI, temperature, soil moisture, and humidity."
    else:
        return "I can help with crop recommendation, yield, irrigation, fertilizer, soil pH, NDVI, and crop stress."

st.markdown("""
<div class="title-box">
    <h1>🌾 AgriTech Smart Assistant</h1>
    <p>All-in-one AI dashboard for crop recommendation, yield, irrigation, fertilizer, pH, NDVI, stress, and more.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Chatbot",
        "Crop Recommendation",
        "Crop Yield",
        "Irrigation",
        "Fertilizer",
        "Soil & pH",
        "Price Prediction",
        "Harvest Time",
        "NPK Prediction",
        "NDVI & Stress",
        "Image Preview",
        "All Models Info",
        "Model Status"
    ]
)

if page == "Home":
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.subheader("🚀 Ultimate Agritech Smart Dashboard")
        st.write("""
This app combines multiple agriculture AI/ML modules in one place.

Available functions:
- Crop recommendation
- Crop yield prediction
- Irrigation prediction
- Fertilizer recommendation
- Soil pH prediction
- Price prediction
- Harvest time prediction
- NPK prediction
- NDVI calculation
- Crop stress detection
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.subheader("🎯 Best For")
        st.write("""
- BSc Agriculture students
- AI/ML agriculture projects
- Final year major projects
- Agritech portfolio building
- Streamlit deployment practice
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Chatbot":
    st.subheader("💬 Agritech Chatbot")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    prompt = st.chat_input("Ask about crop, irrigation, yield, fertilizer, NDVI...")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        reply = bot_response_from_prompt(prompt)
        st.session_state.chat_history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.write(reply)

elif page == "Crop Recommendation":
    st.subheader("🌱 Crop Recommendation")
    c1, c2, c3 = st.columns(3)

    with c1:
        N = safe_float_input("Nitrogen (N)", 90)
        P = safe_float_input("Phosphorus (P)", 42)
        K = safe_float_input("Potassium (K)", 43)

    with c2:
        temperature = safe_float_input("Temperature", 25)
        humidity = safe_float_input("Humidity", 80)

    with c3:
        ph = safe_float_input("pH", 6.5)
        rainfall = safe_float_input("Rainfall", 200)

    if st.button("Predict Crop"):
        result = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall])
        st.success(f"Recommended Crop: {result}")

elif page == "Crop Yield":
    st.subheader("🌾 Crop Yield Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        ph = safe_float_input("pH", 6.5)
        temperature = safe_float_input("Temperature", 26)

    with c2:
        rainfall = safe_float_input("Rainfall", 180)
        fertilizer = safe_float_input("Fertilizer", 120)

    with c3:
        humidity = safe_float_input("Humidity", 75)
        soil_moisture = safe_float_input("Soil Moisture", 40)

    if st.button("Predict Yield"):
        result = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer, humidity, soil_moisture])
        st.success(f"Predicted Yield: {result}")

elif page == "Irrigation":
    st.subheader("💧 Irrigation Prediction")
    c1, c2 = st.columns(2)

    with c1:
        soil_moisture = safe_float_input("Soil Moisture", 35)
        temperature = safe_float_input("Temperature", 28)
        humidity = safe_float_input("Humidity", 70)

    with c2:
        ph = safe_float_input("pH", 6.8)
        rainfall = safe_float_input("Rainfall", 120)

    if st.button("Predict Irrigation"):
        result = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall])
        st.success(f"Irrigation Result: {result}")

elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")
    c1, c2, c3 = st.columns(3)

    with c1:
        N = safe_float_input("Nitrogen (N)", 80)
        P = safe_float_input("Phosphorus (P)", 40)
        K = safe_float_input("Potassium (K)", 40)
        temperature = safe_float_input("Temperature", 27)

    with c2:
        humidity = safe_float_input("Humidity", 70)
        moisture = safe_float_input("Moisture", 30)
        soil_type = safe_float_input("Soil Type (encoded)", 1)

    with c3:
        crop_type = safe_float_input("Crop Type (encoded)", 1)
        ph = safe_float_input("pH", 6.5)
        rainfall = safe_float_input("Rainfall", 150)

    if st.button("Predict Fertilizer"):
        result = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall])
        st.success(f"Recommended Fertilizer: {result}")

elif page == "Soil & pH":
    st.subheader("🌍 Soil pH Prediction")
    soil_moisture = safe_float_input("Soil Moisture", 40)
    organic_matter = safe_float_input("Organic Matter", 2.2)
    temperature = safe_float_input("Temperature", 27)
    rainfall = safe_float_input("Rainfall", 140)

    if st.button("Predict Soil pH"):
        result = agrimodels.predict_ph([soil_moisture, organic_matter, temperature, rainfall])
        st.success(f"Predicted Soil pH: {result}")

elif page == "Price Prediction":
    st.subheader("💰 Crop Price Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        year = safe_float_input("Year", 2026)
        month = safe_float_input("Month", 4)

    with c2:
        market_code = safe_float_input("Market Code", 101)
        arrival_qty = safe_float_input("Arrival Quantity", 200)

    with c3:
        demand_index = safe_float_input("Demand Index", 1.2)
        crop_code = safe_float_input("Crop Code", 10)

    if st.button("Predict Price"):
        result = agrimodels.predict_price([year, month, market_code, arrival_qty, demand_index, crop_code])
        st.success(f"Predicted Crop Price: {result}")

elif page == "Harvest Time":
    st.subheader("🕒 Harvest Time Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        days_after_sowing = safe_float_input("Days After Sowing", 90)
        temperature = safe_float_input("Temperature", 28)

    with c2:
        humidity = safe_float_input("Humidity", 76)
        ph = safe_float_input("pH", 6.4)

    with c3:
        rainfall = safe_float_input("Rainfall", 130)
        soil_moisture = safe_float_input("Soil Moisture", 35)

    if st.button("Predict Harvest Time"):
        result = agrimodels.predict_harvest_time([days_after_sowing, temperature, humidity, ph, rainfall, soil_moisture])
        st.success(f"Predicted Harvest Time: {result}")

elif page == "NPK Prediction":
    st.subheader("🧬 NPK Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        ph = safe_float_input("pH", 6.5)
        ec = safe_float_input("EC", 1.2)

    with c2:
        organic_carbon = safe_float_input("Organic Carbon", 0.8)
        moisture = safe_float_input("Moisture", 30)

    with c3:
        temperature = safe_float_input("Temperature", 27)
        rainfall = safe_float_input("Rainfall", 140)

    if st.button("Predict NPK"):
        result = agrimodels.predict_npk([ph, ec, organic_carbon, moisture, temperature, rainfall])
        st.success(f"NPK Prediction: {result}")

elif page == "NDVI & Stress":
    st.subheader("📈 NDVI and Crop Stress")
    tab1, tab2, tab3 = st.tabs(["NDVI Formula", "NDVI Model", "Stress Prediction"])

    with tab1:
        red = safe_float_input("Red Band", 0.3)
        nir = safe_float_input("NIR Band", 0.7)
        if st.button("Calculate NDVI"):
            ndvi = agrimodels.calculate_ndvi(red, nir)
            st.success(f"NDVI Value: {ndvi:.4f}")

    with tab2:
        red_band = safe_float_input("Red Band Input", 0.3)
        nir_band = safe_float_input("NIR Band Input", 0.7)
        if st.button("Predict NDVI"):
            result = agrimodels.predict_ndvi([red_band, nir_band])
            st.success(f"Predicted NDVI: {result}")

    with tab3:
        ndvi_val = safe_float_input("NDVI", 0.65)
        temperature = safe_float_input("Temperature", 28)
        soil_moisture = safe_float_input("Soil Moisture", 32)
        humidity = safe_float_input("Humidity", 72)
        if st.button("Predict Crop Stress"):
            result = agrimodels.predict_crop_stress([ndvi_val, temperature, soil_moisture, humidity])
            st.success(f"Crop Stress Status: {result}")

elif page == "Image Preview":
    st.subheader("🖼️ Image Preview")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image, image_array = preprocess_uploaded_image(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        st.info(f"Image array shape: {image_array.shape}")

elif page == "All Models Info":
    st.subheader("📚 Available Model Schemas")
    show_model_card("Crop Recommendation", "Features: N, P, K, temperature, humidity, pH, rainfall")
    show_model_card("Crop Yield", "Features: pH, temperature, rainfall, fertilizer, humidity, soil_moisture")
    show_model_card("Irrigation", "Features: soil_moisture, temperature, humidity, pH, rainfall")
    show_model_card("Fertilizer", "Features: N, P, K, temperature, humidity, moisture, soil_type, crop_type, pH, rainfall")
    show_model_card("Soil pH", "Features: soil_moisture, organic_matter, temperature, rainfall")
    show_model_card("Price", "Features: year, month, market_code, arrival_qty, demand_index, crop_code")
    show_model_card("Harvest", "Features: days_after_sowing, temperature, humidity, pH, rainfall, soil_moisture")
    show_model_card("NPK", "Features: pH, EC, organic_carbon, moisture, temperature, rainfall")
    show_model_card("NDVI", "Features: red_band, nir_band")
    show_model_card("Stress", "Features: NDVI, temperature, soil_moisture, humidity")

elif page == "Model Status":
    st.subheader("🧾 Model Status")
    status = agrimodels.get_status()

    for key, value in status.items():
        if value.startswith("Loaded"):
            st.success(f"{key}: {value}")
        elif value.startswith("Missing"):
            st.warning(f"{key}: {value}")
        else:
            st.error(f"{key}: {value}")