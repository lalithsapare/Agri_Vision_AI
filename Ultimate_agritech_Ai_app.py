# Cell 3: Streamlit app (save as app.py)

import streamlit as st
import numpy as np
import os
from PIL import Image

@st.cache_resource
def load_agritech_models():
    models_dir = r"C:\Users\Admin\Agritech_models"   # change this path
    agrimodels = AgritechModels(models_dir)
    return agrimodels

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
}
.title-box {
    background: linear-gradient(90deg, #1b7f5b, #35a37b);
    padding: 18px;
    border-radius: 16px;
    color: white;
    box-shadow: 0 8px 18px rgba(53,163,123,0.25);
}
.small-note {
    font-size: 14px;
    color: #4f6f63;
}
.chat-user {
    background: #dff6e8;
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 8px;
}
.chat-bot {
    background: #ffffff;
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 8px;
    border: 1px solid #dcefe5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🌾 AgriTech Smart Assistant</h1>
    <p>AI-powered agriculture prediction dashboard with chatbot, image analysis, and smart model integration.</p>
</div>
""", unsafe_allow_html=True)

try:
    agrimodels = load_agritech_models()
    st.sidebar.success("✅ Agritech models loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model loading error: {e}")
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
        return "Go to the Crop Recommendation panel and enter N, P, K, temperature, humidity, pH, and rainfall."
    elif "yield" in text:
        return "Go to the Crop Yield panel and enter pH, temperature, rainfall, fertilizer, humidity, and soil moisture."
    elif "irrigation" in text:
        return "Use the Irrigation panel to predict irrigation need from soil moisture, temperature, humidity, pH, and rainfall."
    elif "disease" in text:
        return "Upload a plant image in the Image Models section and choose Plant Disease model."
    elif "ndvi" in text:
        return "Use the NDVI Calculator section with Red and NIR band values."
    elif "npk" in text:
        return "Use the NPK Prediction section with pH, EC, organic carbon, moisture, temperature, and rainfall."
    else:
        return "I can help with crop recommendation, yield, irrigation, disease, weed, soil, nutrient deficiency, NDVI, stress, and NPK prediction."

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

if page == "Home":
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.subheader("🚀 Ultimate Agritech Integration App")
        st.write("""
This app combines multiple AI and ML agriculture models into one smart dashboard.
You can use it for:
- Crop recommendation
- Yield prediction
- Irrigation prediction
- Fertilizer recommendation
- Rainfall / humidity / temperature prediction
- Soil pH and nutrient analysis
- NDVI and crop stress
- Plant disease and soil image classification
- Smart chatbot guidance
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="agri-card">', unsafe_allow_html=True)
        st.subheader("📌 Beginner Friendly")
        st.write("""
Designed for:
- Final year projects
- Portfolio projects
- Agritech startup demos
- AI/ML deployment learning
- Streamlit cloud deployment
        """)
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

elif page == "Weather Models":
    st.subheader("⛅ Weather Models")

    tab1, tab2, tab3 = st.tabs(["Temperature", "Rainfall", "Humidity"])

    with tab1:
        s1 = safe_float_input("Sensor 1", 25)
        s2 = safe_float_input("Sensor 2", 27)
        if st.button("Predict Temperature"):
            result = agrimodels.predict_temperature([s1, s2])
            st.success(f"Predicted Temperature: {result}")

    with tab2:
        t = safe_float_input("Temperature", 29)
        h = safe_float_input("Humidity", 75)
        p = safe_float_input("Pressure", 1012)
        w = safe_float_input("Wind Speed", 10)
        if st.button("Predict Rainfall"):
            result = agrimodels.predict_rainfall([t, h, p, w])
            st.success(f"Predicted Rainfall: {result}")

    with tab3:
        t2 = safe_float_input("Temperature ", 29)
        p2 = safe_float_input("Pressure ", 1011)
        w2 = safe_float_input("Wind Speed ", 8)
        if st.button("Predict Humidity"):
            result = agrimodels.predict_humidity([t2, p2, w2])
            st.success(f"Predicted Humidity: {result}")

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
        st.json(result)

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

    show_model_card("Crop Recommendation", "Features: N, P, K, temperature, humidity, ph, rainfall")
    show_model_card("Crop Yield", "Features: ph, temperature, rainfall, fertilizer, humidity, soil_moisture")
    show_model_card("Irrigation", "Features: soil_moisture, temperature, humidity, ph, rainfall")
    show_model_card("Fertilizer", "Features: N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall")
    show_model_card("Temperature", "Features: sensor_1, sensor_2")
    show_model_card("Rainfall", "Features: temperature, humidity, pressure, wind_speed")
    show_model_card("Humidity", "Features: temperature, pressure, wind_speed")
    show_model_card("pH", "Features: soil_moisture, organic_matter, temperature, rainfall")
    show_model_card("Price", "Features: year, month, market_code, arrival_qty, demand_index, crop_code")
    show_model_card("Harvest", "Features: days_after_sowing, temperature, humidity, ph, rainfall, soil_moisture")
    show_model_card("NPK", "Features: ph, ec, organic_carbon, moisture, temperature, rainfall")
    show_model_card("NDVI", "Features: red_band, nir_band")
    show_model_card("Stress", "Features: ndvi, temperature, soil_moisture, humidity")
    show_model_card("Image Models", "Disease, leaf, weed, soil, nutrient deficiency image classifiers")