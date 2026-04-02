import os
import requests
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
.metric-container {
    background: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
}
.alert-high {
    background:#fee2e2;
    color:#991b1b;
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_df" not in st.session_state:
    st.session_state.latest_df = None

if "farm_data" not in st.session_state:
    st.session_state.farm_data = {}

def get_secret(name, default=""):
    try:
        if name in st.secrets:
            val = str(st.secrets[name]).strip()
            if val:
                return val
    except Exception:
        pass
    env_val = os.getenv(name, "").strip()
    if env_val:
        return env_val
    return default

def fetch_weather(city):
    api_key = get_secret("OPENWEATHER_API_KEY")
    debug = {
        "city": city,
        "api_key_present": bool(api_key),
        "status": "unknown",
        "reason": "",
        "fallback_used": False
    }

    if not api_key:
        debug["status"] = "missing_api_key"
        debug["reason"] = "OPENWEATHER_API_KEY not found"
        debug["fallback_used"] = True
        return None, debug

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=20)
        debug["http_status"] = r.status_code

        if r.status_code != 200:
            debug["status"] = "api_error"
            try:
                debug["reason"] = r.json()
            except Exception:
                debug["reason"] = r.text
            debug["fallback_used"] = True
            return None, debug

        data = r.json()
        if "main" not in data or "weather" not in data:
            debug["status"] = "invalid_response"
            debug["reason"] = data
            debug["fallback_used"] = True
            return None, debug

        result = {
            "temperature": float(data["main"].get("temp", 28.0)),
            "humidity": float(data["main"].get("humidity", 70.0)),
            "rainfall": float(data.get("rain", {}).get("1h", 0.0)),
            "weather": data["weather"][0].get("main", "Unknown"),
            "city": data.get("name", city)
        }
        debug["status"] = "success"
        return result, debug

    except requests.exceptions.Timeout:
        debug["status"] = "timeout"
        debug["reason"] = "Request timed out"
        debug["fallback_used"] = True
        return None, debug

    except requests.exceptions.ConnectionError:
        debug["status"] = "connection_error"
        debug["reason"] = "Network connection failed"
        debug["fallback_used"] = True
        return None, debug

    except Exception as e:
        debug["status"] = "exception"
        debug["reason"] = str(e)
        debug["fallback_used"] = True
        return None, debug

def crop_score(crop, temp, humidity, rainfall, ph, moisture, n, p, k):
    score = 50.0

    rules = {
        "Rice":      [(20 <= temp <= 35, 14), (humidity >= 65, 10), (rainfall >= 80, 12), (moisture >= 40, 10), (5.5 <= ph <= 7.5, 7)],
        "Cotton":    [(25 <= temp <= 35, 15), (40 <= humidity <= 80, 10), (rainfall >= 40, 10), (6.0 <= ph <= 8.0, 8), (k >= 40, 5)],
        "Maize":     [(20 <= temp <= 32, 15), (humidity >= 50, 8), (rainfall >= 30, 10), (5.5 <= ph <= 7.5, 8), (n >= 60, 7)],
        "Wheat":     [(15 <= temp <= 25, 15), (humidity >= 40, 7), (rainfall >= 20, 8), (6.0 <= ph <= 7.5, 8)],
        "Sugarcane": [(24 <= temp <= 35, 15), (humidity >= 55, 8), (rainfall >= 60, 10), (moisture >= 35, 10), (6.0 <= ph <= 8.0, 7)],
        "Soybean":   [(20 <= temp <= 30, 14), (humidity >= 55, 8), (rainfall >= 50, 10), (6.0 <= ph <= 7.5, 8)],
        "Groundnut": [(22 <= temp <= 32, 14), (rainfall >= 25, 8), (6.0 <= ph <= 7.5, 8), (k >= 35, 5)],
        "Chilli":    [(20 <= temp <= 30, 14), (humidity >= 45, 7), (rainfall >= 20, 8), (6.0 <= ph <= 7.5, 8)],
        "Turmeric":  [(20 <= temp <= 32, 14), (humidity >= 60, 10), (rainfall >= 60, 10), (moisture >= 35, 8)]
    }

    if crop in rules:
        for condition, add_score in rules[crop]:
            if condition:
                score += add_score

    score += min(n / 20, 5)
    score += min(p / 20, 5)
    score += min(k / 20, 5)

    return round(min(score, 98), 1)

def predict_yield(crop, temp, humidity, rainfall, ph, moisture, n, p, k):
    base = {
        "Rice": 5.5,
        "Cotton": 2.2,
        "Maize": 4.5,
        "Wheat": 4.0,
        "Sugarcane": 8.0,
        "Soybean": 2.0,
        "Groundnut": 2.4,
        "Chilli": 3.0,
        "Turmeric": 6.5
    }.get(crop, 2.0)

    factor = (
        (temp * 0.02) +
        (humidity * 0.01) +
        (rainfall * 0.004) +
        (moisture * 0.02) +
        ((n + p + k) * 0.002) -
        (abs(ph - 6.8) * 0.15)
    )

    return round(max(0.8, base + factor), 2)

def predict_risk(yield_pred):
    return "High" if yield_pred < 4 else "Low"

def irrigation_advice(moisture, temp, rainfall):
    if rainfall > 20 or moisture > 60:
        return "Low"
    if moisture < 30 and temp > 32:
        return "High"
    return "Moderate"

st.sidebar.title("🌾 Agri Vision AI")

district = st.sidebar.selectbox(
    "Select District / City",
    ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Mahabubnagar"]
)

st.sidebar.subheader("🌾 Crop Selection")
crop_list = [
    "Rice", "Cotton", "Maize", "Wheat", "Sugarcane",
    "Soybean", "Groundnut", "Chilli", "Turmeric"
]

selected_crops = st.sidebar.multiselect(
    "Select crops to analyze",
    crop_list,
    default=["Rice"]
)

st.sidebar.markdown("### Manual Backup Inputs")
manual_temp = st.sidebar.number_input("Temperature (°C)", value=29.0)
manual_humidity = st.sidebar.number_input("Humidity (%)", value=70.0)
manual_rainfall = st.sidebar.number_input("Rainfall (mm)", value=8.0)
manual_ph = st.sidebar.number_input("Soil pH", value=6.8)
manual_moisture = st.sidebar.number_input("Soil Moisture (%)", value=38.0)
manual_n = st.sidebar.number_input("Nitrogen (N)", value=80.0)
manual_p = st.sidebar.number_input("Phosphorus (P)", value=42.0)
manual_k = st.sidebar.number_input("Potassium (K)", value=40.0)

st.markdown("""
<div class="hero">
    <h1>Agri Vision AI</h1>
    <p>Multi-crop analysis dashboard with weather, yield, risk, comparison charts, alerts, and chatbot context</p>
</div>
""", unsafe_allow_html=True)

weather_data, weather_debug = fetch_weather(district)

if weather_data is not None:
    temp = weather_data["temperature"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]
    weather_mode = "Live Weather API"
else:
    temp = manual_temp
    humidity = manual_humidity
    rainfall = manual_rainfall
    weather_mode = "Manual backup"

ph = manual_ph
moisture = manual_moisture
n = manual_n
p = manual_p
k = manual_k

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Weather Source", weather_mode)
with m2:
    st.metric("Temperature", f"{temp} °C")
with m3:
    st.metric("Humidity", f"{humidity} %")
with m4:
    st.metric("Rainfall", f"{rainfall} mm")

with st.expander("Weather debug details", expanded=False):
    st.json(weather_debug)

if st.button("🚀 Analyze Selected Crops", type="primary", use_container_width=True):
    if not selected_crops:
        st.warning("Please select at least one crop.")
    else:
        results = []

        for crop in selected_crops:
            yield_pred = predict_yield(crop, temp, humidity, rainfall, ph, moisture, n, p, k)
            risk = predict_risk(yield_pred)
            suitability = crop_score(crop, temp, humidity, rainfall, ph, moisture, n, p, k)
            irrigation = irrigation_advice(moisture, temp, rainfall)

            results.append({
                "Crop": crop,
                "Predicted Yield": yield_pred,
                "Suitability": suitability,
                "Risk": risk,
                "Irrigation": irrigation
            })

        df = pd.DataFrame(results)
        st.session_state.latest_df = df
        st.session_state.farm_data = {
            "selected_crops": selected_crops,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "ph": ph,
            "moisture": moisture,
            "n": n,
            "p": p,
            "k": k
        }

if st.session_state.latest_df is not None:
    df = st.session_state.latest_df

    st.subheader("Multi-Crop Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("🌾 Crop Yield Comparison")
    fig = px.bar(
        df,
        x="Crop",
        y="Predicted Yield",
        color="Risk",
        title="🌾 Crop Yield Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crop KPI Cards")
    cols = st.columns(len(df)) if len(df) <= 4 else st.columns(4)

    for i, row in df.iterrows():
        col = cols[i % len(cols)]
        with col:
            st.metric(
                label=row["Crop"],
                value=f"{row['Predicted Yield']} t/ha",
                delta=row["Risk"]
            )

    st.subheader("Smart Alerts")
    for _, row in df.iterrows():
        if row["Risk"] == "High":
            st.error(f"🚨 {row['Crop']} is at HIGH risk")
        else:
            st.success(f"✅ {row['Crop']} is stable")

    best_crop = df.sort_values("Predicted Yield", ascending=False).iloc[0]
    st.success(f"🏆 Best Crop: {best_crop['Crop']} with {best_crop['Predicted Yield']} t/ha")

st.subheader("AI Chatbot Context Demo")
user_question = st.text_input("Ask for crop advice")

if st.button("Ask AI", use_container_width=True):
    farm_data = st.session_state.get("farm_data", {})
    farm_data["selected_crops"] = selected_crops

    prompt = f"""
Farmer selected crops: {selected_crops}

Weather:
- Temperature: {farm_data.get('temperature', temp)}
- Humidity: {farm_data.get('humidity', humidity)}
- Rainfall: {farm_data.get('rainfall', rainfall)}

Soil:
- pH: {farm_data.get('ph', ph)}
- Moisture: {farm_data.get('moisture', moisture)}
- N: {farm_data.get('n', n)}
- P: {farm_data.get('p', p)}
- K: {farm_data.get('k', k)}

User question:
{user_question}

Give advice for each crop separately:
- Yield improvement
- Irrigation
- Fertilizer
"""

    st.text_area("Prompt sent to chatbot", value=prompt, height=280)

st.markdown("<div class='small-muted'>Agri Vision AI | Multi-crop selection | Comparison charts | Smart alerts | Chatbot-ready context</div>", unsafe_allow_html=True)
