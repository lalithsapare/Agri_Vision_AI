import os
import io
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Agri Vision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #08111f 0%, #020817 100%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #166534 0%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}

.hero {
    background: linear-gradient(90deg, #166534, #22c55e);
    color: white;
    padding: 24px;
    border-radius: 20px;
    margin-bottom: 18px;
}

.card {
    background: rgba(255,255,255,0.97);
    padding: 16px;
    border-radius: 18px;
    margin-bottom: 14px;
}

.section-title {
    color: white;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 8px;
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

.suggestion-btn {
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_df" not in st.session_state:
    st.session_state.latest_df = None

if "farm_data" not in st.session_state:
    st.session_state.farm_data = {}

# =========================
# HELPERS
# =========================
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
        "reason": ""
    }

    if not api_key:
        debug["status"] = "missing_api_key"
        debug["reason"] = "OPENWEATHER_API_KEY not found"
        return None, debug

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        r = requests.get(url, params=params, timeout=20)
        debug["http_status"] = r.status_code

        if r.status_code != 200:
            debug["status"] = "api_error"
            try:
                debug["reason"] = r.json()
            except Exception:
                debug["reason"] = r.text
            return None, debug

        data = r.json()

        if "main" not in data or "weather" not in data:
            debug["status"] = "invalid_response"
            debug["reason"] = data
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
        return None, debug

    except requests.exceptions.ConnectionError:
        debug["status"] = "connection_error"
        debug["reason"] = "Network connection failed"
        return None, debug

    except Exception as e:
        debug["status"] = "exception"
        debug["reason"] = str(e)
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
        for condition, points in rules[crop]:
            if condition:
                score += points

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
    if yield_pred < 3:
        return "High"
    elif yield_pred < 5:
        return "Moderate"
    return "Low"

def estimate_ndvi(moisture, humidity, rainfall):
    ndvi = 0.2 + (moisture * 0.004) + (humidity * 0.002) + (rainfall * 0.001)
    return round(min(0.92, max(0.12, ndvi)), 3)

def generate_ndvi_trend(current_ndvi):
    trend = []
    start = current_ndvi - 0.08
    for i in range(1, 8):
        val = round(min(0.95, max(0.1, start + i * 0.01)), 3)
        trend.append({"Day": f"Day {i}", "NDVI": val})
    return pd.DataFrame(trend)

def irrigation_advice(moisture, temp, rainfall):
    if rainfall > 20 or moisture > 60:
        return "Low"
    elif moisture < 30 and temp > 32:
        return "High"
    return "Moderate"

def generate_farm_report(df, farm_data):
    lines = []
    lines.append("AGRI VISION AI - FARM REPORT")
    lines.append("=" * 40)
    lines.append(f"City: {farm_data.get('district', 'N/A')}")
    lines.append(f"Temperature: {farm_data.get('temperature', 'N/A')} °C")
    lines.append(f"Humidity: {farm_data.get('humidity', 'N/A')} %")
    lines.append(f"Rainfall: {farm_data.get('rainfall', 'N/A')} mm")
    lines.append(f"Soil pH: {farm_data.get('ph', 'N/A')}")
    lines.append(f"Soil Moisture: {farm_data.get('moisture', 'N/A')} %")
    lines.append(f"N: {farm_data.get('n', 'N/A')}")
    lines.append(f"P: {farm_data.get('p', 'N/A')}")
    lines.append(f"K: {farm_data.get('k', 'N/A')}")
    lines.append("")
    lines.append("SELECTED CROPS ANALYSIS")
    lines.append(df.to_string(index=False))
    lines.append("")
    best_crop = df.sort_values("Predicted Yield", ascending=False).iloc[0]
    lines.append(f"Best Crop: {best_crop['Crop']} with {best_crop['Predicted Yield']} t/ha")
    return "\n".join(lines)

def structured_chat_reply(question, farm_data, df):
    selected_crops = farm_data.get("selected_crops", [])
    best_crop = "Not available"

    if df is not None and not df.empty:
        best_crop = df.sort_values("Predicted Yield", ascending=False).iloc[0]["Crop"]

    return f"""
### Situation
- Selected crops: {', '.join(selected_crops) if selected_crops else 'No crops selected'}
- District: {farm_data.get('district', 'N/A')}
- Temperature: {farm_data.get('temperature', 'N/A')} °C
- Humidity: {farm_data.get('humidity', 'N/A')} %
- Rainfall: {farm_data.get('rainfall', 'N/A')} mm
- Soil pH: {farm_data.get('ph', 'N/A')}
- Soil moisture: {farm_data.get('moisture', 'N/A')} %
- NDVI: {farm_data.get('ndvi', 'N/A')}

### Recommendation
- Best crop right now: {best_crop}
- Monitor moisture and NDVI together.
- Prioritize irrigation if temperature rises and moisture drops.
- Maintain balanced NPK for consistent yield.

### Next Actions
- Compare crop yields in the chart.
- Check smart alerts.
- Download the farm report for documentation.

### Reply
- {question}
- Based on your current farm conditions, focus on the crop with the highest predicted yield and lowest risk.
""".strip()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌾 Agri Vision AI")

district = st.sidebar.selectbox(
    "Select District / City",
    ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Mahabubnagar"]
)

crop_list = [
    "Rice", "Cotton", "Maize", "Wheat", "Sugarcane",
    "Soybean", "Groundnut", "Chilli", "Turmeric"
]

selected_crops = st.sidebar.multiselect(
    "Select crops to analyze",
    crop_list,
    default=["Rice"]
)

ph = st.sidebar.number_input("Soil pH", value=6.8)
moisture = st.sidebar.number_input("Soil Moisture (%)", value=38.0)
n = st.sidebar.number_input("Nitrogen (N)", value=80.0)
p = st.sidebar.number_input("Phosphorus (P)", value=42.0)
k = st.sidebar.number_input("Potassium (K)", value=40.0)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <h1>Agri Vision AI</h1>
    <p>Multi-crop analysis, auto weather, smart alerts, farm report download, advanced charts, and chatbot memory</p>
</div>
""", unsafe_allow_html=True)

# =========================
# WEATHER
# =========================
weather_data, weather_debug = fetch_weather(district)

if weather_data is not None:
    temp = weather_data["temperature"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]
    weather_mode = "Live Weather API"
else:
    temp = 29.0
    humidity = 70.0
    rainfall = 8.0
    weather_mode = "Fallback defaults"

w1, w2, w3, w4 = st.columns(4)
with w1:
    st.metric("Weather Source", weather_mode)
with w2:
    st.metric("Temperature", f"{temp} °C")
with w3:
    st.metric("Humidity", f"{humidity} %")
with w4:
    st.metric("Rainfall", f"{rainfall} mm")

with st.expander("Weather debug details", expanded=False):
    st.json(weather_debug)

# =========================
# ANALYSIS
# =========================
if st.button("🚀 Analyze Selected Crops", type="primary", use_container_width=True):
    if not selected_crops:
        st.warning("Please select at least one crop.")
    else:
        results = []
        ndvi_val = estimate_ndvi(moisture, humidity, rainfall)

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
                "Irrigation": irrigation,
                "NDVI": ndvi_val
            })

        df = pd.DataFrame(results)
        st.session_state.latest_df = df
        st.session_state.farm_data = {
            "selected_crops": selected_crops,
            "district": district,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "ph": ph,
            "moisture": moisture,
            "n": n,
            "p": p,
            "k": k,
            "ndvi": ndvi_val
        }

# =========================
# RESULTS
# =========================
if st.session_state.latest_df is not None:
    df = st.session_state.latest_df

    st.subheader("Multi-Crop Analysis")
    st.dataframe(df, use_container_width=True)

    best_crop = df.sort_values("Predicted Yield", ascending=False).iloc[0]
    st.success(f"🏆 Best Crop: {best_crop['Crop']} with {best_crop['Predicted Yield']} t/ha")

    st.subheader("Smart Alerts")
    ndvi_now = st.session_state.farm_data.get("ndvi", 0.3)

    for _, row in df.iterrows():
        if row["Risk"] == "High":
            st.markdown(f"<div class='alert-high'>🚨 {row['Crop']} is at HIGH risk</div>", unsafe_allow_html=True)
        elif row["Risk"] == "Moderate":
            st.markdown(f"<div class='alert-med'>⚠ {row['Crop']} is at MODERATE risk</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-good'>✅ {row['Crop']} is stable</div>", unsafe_allow_html=True)

    if ndvi_now < 0.35:
        st.markdown("<div class='alert-med'>⚠ Low NDVI detected</div>", unsafe_allow_html=True)

    if n < 40 or p < 25 or k < 25:
        st.markdown("<div class='alert-med'>⚠ Soil imbalance detected</div>", unsafe_allow_html=True)

    st.subheader("Advanced Visualization")

    c1, c2 = st.columns(2)

    with c1:
        fig_yield = px.bar(
            df,
            x="Crop",
            y="Predicted Yield",
            color="Risk",
            title="Yield Comparison Graph"
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    with c2:
        ndvi_trend_df = generate_ndvi_trend(ndvi_now)
        fig_ndvi = px.line(
            ndvi_trend_df,
            x="Day",
            y="NDVI",
            markers=True,
            title="NDVI Trend Chart"
        )
        st.plotly_chart(fig_ndvi, use_container_width=True)

    radar_df = pd.DataFrame({
        "Metric": ["Nitrogen", "Phosphorus", "Potassium", "Moisture", "pH x10"],
        "Value": [n, p, k, moisture, ph * 10]
    })

    fig_radar = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        line_close=True,
        title="Soil Radar Chart"
    )
    fig_radar.update_traces(fill="toself")
    st.plotly_chart(fig_radar, use_container_width=True)

    report_text = generate_farm_report(df, st.session_state.farm_data)
    st.download_button(
        "Download Farm Report",
        data=report_text,
        file_name="farm_report.txt",
        mime="text/plain"
    )

# =========================
# CHATBOT
# =========================
st.subheader("AI Assistant")

s1, s2, s3 = st.columns(3)
with s1:
    if st.button("Suggest best crop"):
        st.session_state.chat_history.append({"role": "user", "content": "Suggest best crop"})
with s2:
    if st.button("Irrigation advice"):
        st.session_state.chat_history.append({"role": "user", "content": "Give irrigation advice"})
with s3:
    if st.button("Fertilizer advice"):
        st.session_state.chat_history.append({"role": "user", "content": "Give fertilizer advice"})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask about yield, crops, irrigation, NDVI, soil, or alerts")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "user":
        reply = structured_chat_reply(
            last_msg["content"],
            st.session_state.farm_data,
            st.session_state.latest_df
        )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

st.markdown(
    "<div class='small-muted'>Agri Vision AI | Multi-crop + live weather + report download + smart alerts + advanced charts + chatbot memory</div>",
    unsafe_allow_html=True
)
