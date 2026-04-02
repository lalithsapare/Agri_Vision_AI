import os
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Agri Vision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #08111f 0%, #020817 100%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #166534 0%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero {
    background: linear-gradient(90deg, #166534, #22c55e);
    color: white;
    padding: 22px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.card {
    background: rgba(255,255,255,0.97);
    padding: 16px;
    border-radius: 18px;
    margin-bottom: 12px;
}
.alert-high {
    background:#fee2e2;
    color:#991b1b;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
    font-weight:600;
}
.alert-med {
    background:#fef3c7;
    color:#92400e;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
    font-weight:600;
}
.alert-good {
    background:#dcfce7;
    color:#166534;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
    font-weight:600;
}
.small-muted {
    color:#64748b;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

ALL_CROPS = [
    "Cotton",
    "Maize",
    "Red Gram",
    "Soybean",
    "Turmeric",
    "Jowar",
    "Paddy",
    "Groundnut",
    "Black Gram",
    "Bengal Gram",
    "Sesame",
    "Sunflower"
]

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

def get_secret(name, default=""):
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:
        pass
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    return default

def fetch_weather(city):
    api_key = get_secret("OPENWEATHER_API_KEY")
    debug = {"city": city, "has_key": bool(api_key), "status": "unknown", "reason": ""}

    if not api_key:
        debug["status"] = "missing_key"
        debug["reason"] = "OPENWEATHER_API_KEY not found"
        return None, debug

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(url, params=params, timeout=20)
        debug["http_status"] = response.status_code

        if response.status_code != 200:
            debug["status"] = "api_error"
            try:
                debug["reason"] = response.json()
            except Exception:
                debug["reason"] = response.text
            return None, debug

        data = response.json()
        result = {
            "temperature": float(data["main"].get("temp", 28.0)),
            "humidity": float(data["main"].get("humidity", 70.0)),
            "rainfall": float(data.get("rain", {}).get("1h", 0.0)),
            "weather": data["weather"][0]["main"] if data.get("weather") else "Unknown",
            "city": data.get("name", city)
        }
        debug["status"] = "success"
        return result, debug

    except Exception as e:
        debug["status"] = "exception"
        debug["reason"] = str(e)
        return None, debug

def crop_rules(crop, temp, humidity, rainfall, ph, moisture, n, p, k):
    score = 50

    if crop == "Cotton":
        if 25 <= temp <= 35: score += 15
        if 40 <= humidity <= 80: score += 10
        if rainfall >= 40: score += 10
        if 6.0 <= ph <= 8.0: score += 8
        if k >= 40: score += 5

    elif crop == "Maize":
        if 20 <= temp <= 32: score += 15
        if humidity >= 50: score += 8
        if rainfall >= 30: score += 10
        if 5.5 <= ph <= 7.5: score += 8
        if n >= 60: score += 7

    elif crop == "Red Gram":
        if 22 <= temp <= 34: score += 14
        if rainfall >= 20: score += 10
        if 6.0 <= ph <= 7.5: score += 8
        if moisture >= 25: score += 6

    elif crop == "Soybean":
        if 20 <= temp <= 30: score += 14
        if humidity >= 55: score += 8
        if rainfall >= 50: score += 10
        if 6.0 <= ph <= 7.5: score += 8

    elif crop == "Turmeric":
        if 20 <= temp <= 32: score += 14
        if humidity >= 60: score += 10
        if rainfall >= 60: score += 10
        if moisture >= 35: score += 8

    elif crop == "Jowar":
        if 24 <= temp <= 36: score += 14
        if rainfall >= 20: score += 8
        if 5.5 <= ph <= 8.0: score += 8
        if moisture >= 20: score += 5

    elif crop == "Paddy":
        if 20 <= temp <= 35: score += 14
        if humidity >= 65: score += 10
        if rainfall >= 80: score += 12
        if moisture >= 40: score += 10
        if 5.5 <= ph <= 7.5: score += 7

    elif crop == "Groundnut":
        if 22 <= temp <= 32: score += 14
        if rainfall >= 25: score += 8
        if 6.0 <= ph <= 7.5: score += 8
        if calcium_like(k) >= 35: score += 5

    elif crop == "Black Gram":
        if 24 <= temp <= 34: score += 12
        if rainfall >= 20: score += 8
        if humidity >= 45: score += 7
        if 6.0 <= ph <= 7.5: score += 8

    elif crop == "Bengal Gram":
        if 18 <= temp <= 30: score += 14
        if rainfall <= 60: score += 10
        if 6.0 <= ph <= 8.0: score += 8
        if moisture >= 20: score += 5

    elif crop == "Sesame":
        if 24 <= temp <= 34: score += 13
        if rainfall >= 20: score += 8
        if 5.5 <= ph <= 7.5: score += 8

    elif crop == "Sunflower":
        if 20 <= temp <= 32: score += 13
        if rainfall >= 20: score += 8
        if humidity <= 75: score += 7
        if 6.0 <= ph <= 7.8: score += 8

    score += min(n / 20, 5)
    score += min(p / 20, 5)
    score += min(k / 20, 5)

    return round(min(score, 98), 1)

def calcium_like(val):
    return val

def predict_yield(crop, temp, humidity, rainfall, ph, moisture, n, p, k):
    base = {
        "Cotton": 22,
        "Maize": 45,
        "Red Gram": 14,
        "Soybean": 20,
        "Turmeric": 65,
        "Jowar": 18,
        "Paddy": 55,
        "Groundnut": 24,
        "Black Gram": 13,
        "Bengal Gram": 16,
        "Sesame": 10,
        "Sunflower": 17
    }.get(crop, 20)

    factor = (
        (temp * 0.18) +
        (humidity * 0.07) +
        (rainfall * 0.03) +
        (moisture * 0.22) +
        ((n + p + k) * 0.04) -
        (abs(ph - 6.8) * 2.0)
    )

    prediction = max(1.0, round((base + factor) / 10, 2))
    return prediction

def irrigation_advice(moisture, temp, rainfall):
    if rainfall > 20 or moisture > 60:
        return "Low", "Skip irrigation today"
    if moisture < 30 and temp > 32:
        return "High", "Provide deep irrigation immediately"
    return "Moderate", "Light irrigation recommended"

def rain_prediction(rainfall, humidity, temp):
    chance = min(100, max(0, round((humidity * 0.7) + (rainfall * 8) - (temp * 0.4), 1)))
    return chance

def smart_alerts(crop, moisture, rainfall, temp, suitability):
    alerts = []

    if suitability < 65:
        alerts.append(("high", f"🚨 {crop} suitability is low for current conditions."))

    if moisture < 25:
        alerts.append(("med", "⚠ Low soil moisture detected."))

    if rainfall < 5 and temp > 35:
        alerts.append(("high", "🚨 High heat and low rain risk."))

    if 25 <= moisture <= 60 and suitability >= 75:
        alerts.append(("good", f"✅ {crop} is in a reasonably favorable range."))

    return alerts

st.sidebar.title("🌾 Agri Vision AI")
district = st.sidebar.selectbox(
    "Select District / City",
    ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Mahabubnagar"]
)
selected_crop = st.sidebar.selectbox("Choose Crop", ALL_CROPS)

st.sidebar.markdown("### Backup Manual Inputs")
with st.sidebar.expander("Use only when weather API fails", expanded=False):
    manual_temp = st.number_input("Manual Temperature (°C)", value=29.0)
    manual_humidity = st.number_input("Manual Humidity (%)", value=70.0)
    manual_rainfall = st.number_input("Manual Rainfall (mm)", value=8.0)
    manual_ph = st.number_input("Soil pH", value=6.8)
    manual_moisture = st.number_input("Soil Moisture (%)", value=38.0)
    manual_n = st.number_input("Nitrogen (N)", value=80.0)
    manual_p = st.number_input("Phosphorus (P)", value=42.0)
    manual_k = st.number_input("Potassium (K)", value=40.0)

st.markdown(f"""
<div class="hero">
    <h1>Agri Vision AI</h1>
    <p>Selected crop: <b>{selected_crop}</b> | City: <b>{district}</b> | All-in-one prediction dashboard</p>
</div>
""", unsafe_allow_html=True)

weather_data, weather_debug = fetch_weather(district)

if weather_data:
    temp = weather_data["temperature"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]
    weather_label = weather_data["weather"]
    weather_mode = "Live API"
else:
    temp = manual_temp
    humidity = manual_humidity
    rainfall = manual_rainfall
    weather_label = "Manual backup"
    weather_mode = "Manual backup"

ph = manual_ph
moisture = manual_moisture
n = manual_n
p = manual_p
k = manual_k

col1, col2, col3, col4 = st.columns(4)
col1.metric("Weather Mode", weather_mode)
col2.metric("Temperature", f"{temp} °C")
col3.metric("Humidity", f"{humidity} %")
col4.metric("Rainfall", f"{rainfall} mm")

if not weather_data:
    st.warning("Weather API unavailable. Manual backup inputs enabled.")

with st.expander("Weather debug details", expanded=False):
    st.json(weather_debug)

if st.button("🚀 Predict All", type="primary", use_container_width=True):
    suitability = crop_rules(selected_crop, temp, humidity, rainfall, ph, moisture, n, p, k)
    yield_pred = predict_yield(selected_crop, temp, humidity, rainfall, ph, moisture, n, p, k)
    irrigation_level, irrigation_action = irrigation_advice(moisture, temp, rainfall)
    rain_chance = rain_prediction(rainfall, humidity, temp)

    result = {
        "crop": selected_crop,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "ph": ph,
        "moisture": moisture,
        "n": n,
        "p": p,
        "k": k,
        "weather_label": weather_label,
        "suitability": suitability,
        "yield_pred": yield_pred,
        "irrigation_level": irrigation_level,
        "irrigation_action": irrigation_action,
        "rain_chance": rain_chance
    }
    st.session_state.latest_result = result

if st.session_state.latest_result:
    result = st.session_state.latest_result

    a, b, c, d = st.columns(4)
    a.metric("Selected Crop", result["crop"])
    b.metric("Suitability", f'{result["suitability"]}%')
    c.metric("Yield Prediction", f'{result["yield_pred"]} t/ha')
    d.metric("Rain Chance", f'{result["rain_chance"]}%')

    x1, x2 = st.columns(2)

    with x1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Summary")
        st.write(f"Weather source: {weather_mode}")
        st.write(f"Current weather: {result['weather_label']}")
        st.write(f"Temperature: {result['temperature']} °C")
        st.write(f"Humidity: {result['humidity']} %")
        st.write(f"Rainfall: {result['rainfall']} mm")
        st.write(f"Soil pH: {result['ph']}")
        st.write(f"Soil moisture: {result['moisture']} %")
        st.write(f"Irrigation need: {result['irrigation_level']}")
        st.write(f"Action: {result['irrigation_action']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with x2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Why this result")
        st.write(f"{result['crop']} was evaluated using current weather, rainfall, humidity, soil pH, moisture, and NPK values.")
        st.write("Higher suitability means the selected crop fits present conditions better.")
        st.write("Yield prediction is an estimated productivity score based on crop base potential and environmental conditions.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Smart Alerts")
    alerts = smart_alerts(
        result["crop"],
        result["moisture"],
        result["rainfall"],
        result["temperature"],
        result["suitability"]
    )

    if alerts:
        for level, msg in alerts:
            if level == "high":
                st.markdown(f'<div class="alert-high">{msg}</div>', unsafe_allow_html=True)
            elif level == "med":
                st.markdown(f'<div class="alert-med">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-good">{msg}</div>', unsafe_allow_html=True)

    st.subheader("Analytics")

    c1, c2 = st.columns(2)

    with c1:
        metric_df = pd.DataFrame({
            "Metric": ["Temperature", "Humidity", "Rainfall", "Moisture", "Suitability", "Rain Chance"],
            "Value": [
                result["temperature"],
                result["humidity"],
                result["rainfall"],
                result["moisture"],
                result["suitability"],
                result["rain_chance"]
            ]
        })
        fig_bar = px.bar(metric_df, x="Metric", y="Value", color="Metric", title="All-in-One Prediction Metrics")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["suitability"],
            title={'text': f"{result['crop']} Suitability"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("<div class='footer'>Agri Vision AI | Crop selected from sidebar | One-click prediction enabled</div>", unsafe_allow_html=True)
