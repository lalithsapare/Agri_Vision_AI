import os
import traceback
import joblib
import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except Exception:
    tf = None


class AgritechModels:
    MODEL_SCHEMA = {
        "crop_recommendation": {"file": "crop_recommendation.pkl", "type": "sklearn"},
        "crop_yield": {"file": "crop_yield.pkl", "type": "sklearn"},
        "irrigation": {"file": "irrigation.pkl", "type": "sklearn"},
        "fertilizer": {"file": "fertilizer.pkl", "type": "sklearn"},
        "temperature": {"file": "temperature.pkl", "type": "sklearn"},
        "rainfall": {"file": "rainfall.pkl", "type": "sklearn"},
        "humidity": {"file": "humidity.pkl", "type": "sklearn"},
        "ph": {"file": "ph.pkl", "type": "sklearn"},
        "price": {"file": "price.pkl", "type": "sklearn"},
        "harvest_time": {"file": "harvest_time.pkl", "type": "sklearn"},
        "npk": {"file": "npk.pkl", "type": "sklearn"},
        "ndvi": {"file": "ndvi.pkl", "type": "sklearn"},
        "crop_stress": {"file": "crop_stress.pkl", "type": "sklearn"},
        "disease": {"file": "disease_model.h5", "type": "tensorflow"},
        "leaf": {"file": "leaf_model.h5", "type": "tensorflow"},
        "weed": {"file": "weed_model.h5", "type": "tensorflow"},
        "soil": {"file": "soil_model.h5", "type": "tensorflow"},
        "nutrient": {"file": "nutrient_model.keras", "type": "tensorflow"}
    }

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self._load_models()

    def _load_models(self):
        missing_files = []

        for key, schema in self.MODEL_SCHEMA.items():
            model_file = os.path.join(self.models_dir, schema["file"])

            if not os.path.exists(model_file):
                missing_files.append(schema["file"])
                continue

            try:
                if model_file.endswith(".h5") or model_file.endswith(".keras"):
                    if tf is None:
                        raise ImportError("TensorFlow is required for .h5/.keras models")
                    self.models[key] = tf.keras.models.load_model(model_file)
                else:
                    self.models[key] = joblib.load(model_file)
            except Exception as e:
                raise RuntimeError(f"Failed loading {schema['file']}: {e}")

        if missing_files:
            raise FileNotFoundError(
                "Missing model files in models/: " + ", ".join(missing_files)
            )

    def _predict_tabular(self, model_key, features):
        model = self.models[model_key]
        arr = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)

        if isinstance(pred, np.ndarray):
            if pred.ndim == 0:
                return pred.item()
            if pred.ndim == 1:
                return pred[0].item() if hasattr(pred[0], "item") else pred[0]
            return pred.tolist()

        return pred

    def _predict_image(self, model_key, image_array):
        model = self.models[model_key]
        arr = np.array(image_array, dtype=np.float32)

        if arr.max() > 1:
            arr = arr / 255.0

        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr, verbose=0)
        pred = np.array(pred)

        return {
            "predicted_class_index": int(np.argmax(pred)),
            "confidence": float(np.max(pred))
        }

    def predict_crop_recommendation(self, features):
        return self._predict_tabular("crop_recommendation", features)

    def predict_crop_yield(self, features):
        return self._predict_tabular("crop_yield", features)

    def predict_irrigation(self, features):
        return self._predict_tabular("irrigation", features)

    def predict_fertilizer(self, features):
        return self._predict_tabular("fertilizer", features)

    def predict_temperature(self, features):
        return self._predict_tabular("temperature", features)

    def predict_rainfall(self, features):
        return self._predict_tabular("rainfall", features)

    def predict_humidity(self, features):
        return self._predict_tabular("humidity", features)

    def predict_ph(self, features):
        return self._predict_tabular("ph", features)

    def predict_price(self, features):
        return self._predict_tabular("price", features)

    def predict_harvest_time(self, features):
        return self._predict_tabular("harvest_time", features)

    def predict_npk(self, features):
        result = self._predict_tabular("npk", features)
        return {"NPK_prediction": result}

    def calculate_ndvi(self, red, nir):
        red = float(red)
        nir = float(nir)
        denominator = nir + red
        if denominator == 0:
            return 0.0
        return (nir - red) / denominator

    def predict_ndvi(self, features):
        return self._predict_tabular("ndvi", features)

    def predict_crop_stress(self, features):
        return self._predict_tabular("crop_stress", features)

    def predict_disease(self, image_array):
        return self._predict_image("disease", image_array)

    def predict_leaf(self, image_array):
        return self._predict_image("leaf", image_array)

    def predict_weed(self, image_array):
        return self._predict_image("weed", image_array)

    def predict_soil(self, image_array):
        return self._predict_image("soil", image_array)

    def predict_nutrient(self, image_array):
        return self._predict_image("nutrient", image_array)


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
    padding-top: 1.1rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3d2e 0%, #145a41 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.title-box {
    background: linear-gradient(90deg, #176947, #33a474);
    padding: 20px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 10px 28px rgba(28, 155, 103, 0.28);
    margin-bottom: 1rem;
}
.agri-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border: 1px solid #dff3e7;
    margin-bottom: 1rem;
}
.small-note {
    color: #47675a;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.getcwd(), "models")

@st.cache_resource(show_spinner="Loading agritech models...")
def load_agritech_models():
    return AgritechModels(MODEL_DIR)

def safe_float_input(label, value=0.0, key=None):
    return st.number_input(label, value=float(value), step=0.1, format="%.2f", key=key)

def preprocess_uploaded_image(uploaded_file, size=(224, 224)):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(size)
    image_array = np.array(image, dtype=np.float32)
    return image, image_array

def show_model_card(model_name, desc):
    with st.expander(f"🔧 {model_name}"):
        st.caption(desc)

def bot_response_from_prompt(prompt):
    text = prompt.lower()

    if "crop" in text and "recommend" in text:
        return "Open Crop Recommendation and enter N, P, K, temperature, humidity, pH, and rainfall."
    if "yield" in text:
        return "Open Crop Yield and enter pH, temperature, rainfall, fertilizer, humidity, and soil moisture."
    if "irrigation" in text or "water" in text:
        return "Open Irrigation and enter soil moisture, temperature, humidity, pH, and rainfall."
    if "disease" in text or "leaf" in text:
        return "Open Image Models, upload a plant image, and run plant disease prediction."
    if "ndvi" in text:
        return "Open NDVI & Stress to calculate NDVI from red and NIR values or run the NDVI model."
    if "npk" in text or "nutrient" in text:
        return "Open NPK Prediction and enter pH, EC, organic carbon, moisture, temperature, and rainfall."

    return "I can guide you for crop recommendation, yield, irrigation, fertilizer, price, harvest, NDVI, stress, and image-based disease prediction."

st.markdown("""
<div class="title-box">
    <h1>🌾 AgriTech Smart Assistant</h1>
    <p>AI-powered agriculture prediction dashboard with chatbot, image analysis, and smart model integration.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🌿 Navigation")
    page = st.radio(
        "Select Module",
        [
            "Home",
            "Chatbot",
            "Crop Recommendation",
            "Crop Yield",
            "Irrigation",
            "Fertilizer",
            "Weather Models",
            "Soil & pH",
            "Price Prediction",
            "Harvest Time",
            "NPK Prediction",
            "NDVI & Stress",
            "Image Models",
            "All Models Info"
        ]
    )

try:
    agrimodels = load_agritech_models()
    st.sidebar.success("✅ Models loaded successfully")
    st.sidebar.caption(f"Model folder: {MODEL_DIR}")
except Exception as e:
    st.sidebar.error(f"❌ Model loading error: {e}")
    with st.expander("Deployment troubleshooting", expanded=True):
        st.code("""
1. Keep this app.py in repo root.
2. Create a folder named models in the same repo.
3. Put all .pkl, .h5, and .keras files inside models.
4. Make sure file names in MODEL_SCHEMA exactly match files in models.
5. Add all required packages in requirements.txt.
        """)
        st.code(traceback.format_exc())
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if page == "Home":
    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.subheader("🚀 Integrated Agritech AI Platform")
        st.write(
            "This dashboard combines multiple agriculture AI models for crop recommendation, yield prediction, irrigation planning, fertilizer advice, price prediction, harvest estimation, NDVI analysis, crop stress detection, and image-based diagnosis."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.subheader("📌 Deployment Notes")
        st.write(
            "Use relative model paths and place all trained models inside a models folder for Streamlit Cloud deployment."
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Chatbot":
    st.subheader("💬 Agritech Chatbot Assistant")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    prompt = st.chat_input("Ask about crop, irrigation, NDVI, disease, yield, fertilizer...")
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
        N = safe_float_input("Nitrogen (N)", 90, key="crop_N")
        P = safe_float_input("Phosphorus (P)", 42, key="crop_P")
        K = safe_float_input("Potassium (K)", 43, key="crop_K")
    with c2:
        temperature = safe_float_input("Temperature", 25, key="crop_temp")
        humidity = safe_float_input("Humidity", 80, key="crop_hum")
    with c3:
        ph = safe_float_input("pH", 6.5, key="crop_ph")
        rainfall = safe_float_input("Rainfall", 200, key="crop_rain")

    if st.button("Predict Crop"):
        result = agrimodels.predict_crop_recommendation([N, P, K, temperature, humidity, ph, rainfall])
        st.success(f"Recommended Crop: {result}")

elif page == "Crop Yield":
    st.subheader("🌾 Crop Yield Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        ph = safe_float_input("pH", 6.5, key="yield_ph")
        temperature = safe_float_input("Temperature", 26, key="yield_temp")
    with c2:
        rainfall = safe_float_input("Rainfall", 180, key="yield_rain")
        fertilizer = safe_float_input("Fertilizer", 120, key="yield_fert")
    with c3:
        humidity = safe_float_input("Humidity", 75, key="yield_hum")
        soil_moisture = safe_float_input("Soil Moisture", 40, key="yield_moist")

    if st.button("Predict Yield"):
        result = agrimodels.predict_crop_yield([ph, temperature, rainfall, fertilizer, humidity, soil_moisture])
        st.success(f"Predicted Yield: {result}")

elif page == "Irrigation":
    st.subheader("💧 Irrigation Prediction")

    c1, c2 = st.columns(2)
    with c1:
        soil_moisture = safe_float_input("Soil Moisture", 35, key="irr_moist")
        temperature = safe_float_input("Temperature", 28, key="irr_temp")
        humidity = safe_float_input("Humidity", 70, key="irr_hum")
    with c2:
        ph = safe_float_input("pH", 6.8, key="irr_ph")
        rainfall = safe_float_input("Rainfall", 120, key="irr_rain")

    if st.button("Predict Irrigation"):
        result = agrimodels.predict_irrigation([soil_moisture, temperature, humidity, ph, rainfall])
        st.success(f"Irrigation Result: {result}")

elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")

    c1, c2, c3 = st.columns(3)
    with c1:
        N = safe_float_input("Nitrogen (N)", 80, key="fert_N")
        P = safe_float_input("Phosphorus (P)", 40, key="fert_P")
        K = safe_float_input("Potassium (K)", 40, key="fert_K")
        temperature = safe_float_input("Temperature", 27, key="fert_temp")
    with c2:
        humidity = safe_float_input("Humidity", 70, key="fert_hum")
        moisture = safe_float_input("Moisture", 30, key="fert_moist")
        soil_type = safe_float_input("Soil Type (encoded)", 1, key="fert_soil")
    with c3:
        crop_type = safe_float_input("Crop Type (encoded)", 1, key="fert_crop")
        ph = safe_float_input("pH", 6.5, key="fert_ph")
        rainfall = safe_float_input("Rainfall", 150, key="fert_rain")

    if st.button("Predict Fertilizer"):
        result = agrimodels.predict_fertilizer([N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall])
        st.success(f"Recommended Fertilizer: {result}")

elif page == "Weather Models":
    st.subheader("⛅ Weather Models")

    tab1, tab2, tab3 = st.tabs(["Temperature", "Rainfall", "Humidity"])

    with tab1:
        s1 = safe_float_input("Sensor 1", 25, key="w_s1")
        s2 = safe_float_input("Sensor 2", 27, key="w_s2")
        if st.button("Predict Temperature"):
            result = agrimodels.predict_temperature([s1, s2])
            st.success(f"Predicted Temperature: {result}")

    with tab2:
        t = safe_float_input("Temperature", 29, key="rain_t")
        h = safe_float_input("Humidity", 75, key="rain_h")
        p = safe_float_input("Pressure", 1012, key="rain_p")
        w = safe_float_input("Wind Speed", 10, key="rain_w")
        if st.button("Predict Rainfall"):
            result = agrimodels.predict_rainfall([t, h, p, w])
            st.success(f"Predicted Rainfall: {result}")

    with tab3:
        t2 = safe_float_input("Temperature", 29, key="hum_t")
        p2 = safe_float_input("Pressure", 1011, key="hum_p")
        w2 = safe_float_input("Wind Speed", 8, key="hum_w")
        if st.button("Predict Humidity"):
            result = agrimodels.predict_humidity([t2, p2, w2])
            st.success(f"Predicted Humidity: {result}")

elif page == "Soil & pH":
    st.subheader("🌍 Soil pH Prediction")

    soil_moisture = safe_float_input("Soil Moisture", 40, key="soil_m")
    organic_matter = safe_float_input("Organic Matter", 2.2, key="soil_o")
    temperature = safe_float_input("Temperature", 27, key="soil_t")
    rainfall = safe_float_input("Rainfall", 140, key="soil_r")

    if st.button("Predict Soil pH"):
        result = agrimodels.predict_ph([soil_moisture, organic_matter, temperature, rainfall])
        st.success(f"Predicted Soil pH: {result}")

elif page == "Price Prediction":
    st.subheader("💰 Crop Price Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        year = safe_float_input("Year", 2026, key="price_y")
        month = safe_float_input("Month", 4, key="price_m")
    with c2:
        market_code = safe_float_input("Market Code", 101, key="price_mc")
        arrival_qty = safe_float_input("Arrival Quantity", 200, key="price_aq")
    with c3:
        demand_index = safe_float_input("Demand Index", 1.2, key="price_di")
        crop_code = safe_float_input("Crop Code", 10, key="price_cc")

    if st.button("Predict Price"):
        result = agrimodels.predict_price([year, month, market_code, arrival_qty, demand_index, crop_code])
        st.success(f"Predicted Crop Price: {result}")

elif page == "Harvest Time":
    st.subheader("🕒 Harvest Time Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        days_after_sowing = safe_float_input("Days After Sowing", 90, key="harvest_days")
        temperature = safe_float_input("Temperature", 28, key="harvest_temp")
    with c2:
        humidity = safe_float_input("Humidity", 76, key="harvest_hum")
        ph = safe_float_input("pH", 6.4, key="harvest_ph")
    with c3:
        rainfall = safe_float_input("Rainfall", 130, key="harvest_rain")
        soil_moisture = safe_float_input("Soil Moisture", 35, key="harvest_sm")

    if st.button("Predict Harvest Time"):
        result = agrimodels.predict_harvest_time([days_after_sowing, temperature, humidity, ph, rainfall, soil_moisture])
        st.success(f"Predicted Harvest Time: {result}")

elif page == "NPK Prediction":
    st.subheader("🧬 NPK Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        ph = safe_float_input("pH", 6.5, key="npk_ph")
        ec = safe_float_input("EC", 1.2, key="npk_ec")
    with c2:
        organic_carbon = safe_float_input("Organic Carbon", 0.8, key="npk_oc")
        moisture = safe_float_input("Moisture", 30, key="npk_mo")
    with c3:
        temperature = safe_float_input("Temperature", 27, key="npk_t")
        rainfall = safe_float_input("Rainfall", 140, key="npk_r")

    if st.button("Predict NPK"):
        result = agrimodels.predict_npk([ph, ec, organic_carbon, moisture, temperature, rainfall])
        st.json(result)

elif page == "NDVI & Stress":
    st.subheader("📈 NDVI and Crop Stress")

    tab1, tab2, tab3 = st.tabs(["NDVI Formula", "NDVI Model", "Stress Prediction"])

    with tab1:
        red = safe_float_input("Red Band", 0.3, key="ndvi_red")
        nir = safe_float_input("NIR Band", 0.7, key="ndvi_nir")
        if st.button("Calculate NDVI"):
            ndvi = agrimodels.calculate_ndvi(red, nir)
            st.success(f"NDVI Value: {ndvi:.4f}")

    with tab2:
        red_band = safe_float_input("Red Band Input", 0.3, key="ndvi_red2")
        nir_band = safe_float_input("NIR Band Input", 0.7, key="ndvi_nir2")
        if st.button("Predict NDVI"):
            result = agrimodels.predict_ndvi([red_band, nir_band])
            st.success(f"Predicted NDVI: {result}")

    with tab3:
        ndvi_val = safe_float_input("NDVI", 0.65, key="stress_ndvi")
        temperature = safe_float_input("Temperature", 28, key="stress_temp")
        soil_moisture = safe_float_input("Soil Moisture", 32, key="stress_sm")
        humidity = safe_float_input("Humidity", 72, key="stress_hum")
        if st.button("Predict Crop Stress"):
            result = agrimodels.predict_crop_stress([ndvi_val, temperature, soil_moisture, humidity])
            st.success(f"Crop Stress Status: {result}")

elif page == "Image Models":
    st.subheader("🖼️ Image-Based Predictions")

    image_model = st.selectbox(
        "Choose Image Model",
        [
            "Plant Disease",
            "Leaf Classification",
            "Weed Detection",
            "Soil Classification",
            "Nutrient Deficiency"
        ]
    )
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image, image_array = preprocess_uploaded_image(uploaded_file)
        st.image(image, caption="Uploaded Image", width=280)

        if st.button("Predict from Image"):
            if image_model == "Plant Disease":
                result = agrimodels.predict_disease(image_array)
            elif image_model == "Leaf Classification":
                result = agrimodels.predict_leaf(image_array)
            elif image_model == "Weed Detection":
                result = agrimodels.predict_weed(image_array)
            elif image_model == "Soil Classification":
                result = agrimodels.predict_soil(image_array)
            else:
                result = agrimodels.predict_nutrient(image_array)

            st.json(result)

elif page == "All Models Info":
    st.subheader("📚 Available Model Schemas")

    show_model_card("Crop Recommendation", "Features: N, P, K, temperature, humidity, pH, rainfall")
    show_model_card("Crop Yield", "Features: pH, temperature, rainfall, fertilizer, humidity, soil_moisture")
    show_model_card("Irrigation", "Features: soil_moisture, temperature, humidity, pH, rainfall")
    show_model_card("Fertilizer", "Features: N, P, K, temperature, humidity, moisture, soil_type, crop_type, pH, rainfall")
    show_model_card("Temperature", "Features: sensor_1, sensor_2")
    show_model_card("Rainfall", "Features: temperature, humidity, pressure, wind_speed")
    show_model_card("Humidity", "Features: temperature, pressure, wind_speed")
    show_model_card("pH", "Features: soil_moisture, organic_matter, temperature, rainfall")
    show_model_card("Price", "Features: year, month, market_code, arrival_qty, demand_index, crop_code")
    show_model_card("Harvest", "Features: days_after_sowing, temperature, humidity, pH, rainfall, soil_moisture")
    show_model_card("NPK", "Features: pH, EC, organic_carbon, moisture, temperature, rainfall")
    show_model_card("NDVI", "Features: red_band, nir_band")
    show_model_card("Stress", "Features: ndvi, temperature, soil_moisture, humidity")
    show_model_card("Image Models", "Disease, leaf, weed, soil, nutrient deficiency image classifiers")