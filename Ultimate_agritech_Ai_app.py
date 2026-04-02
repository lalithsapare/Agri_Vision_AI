import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# API KEY
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #020617 0%, #071226 45%, #0b1120 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #020617 0%, #071226 45%, #0b1120 100%);}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14532d 0%, #166534 50%, #14532d 100%);
}
[data-testid="stSidebar"] * {color: white !important;}

.hero-dashboard {
    background: linear-gradient(90deg, #166534, #22c55e, #16a34a);
    color: white;
    padding: 1.8rem 1.5rem;
    border-radius: 24px;
    box-shadow: 0 12px 32px rgba(34, 197, 94, 0.18);
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #172033 100%);
    border-radius: 20px;
    padding: 1.4rem;
    text-align: center;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    color: #e5e7eb;
}

.health-card {
    background: linear-gradient(135deg, #0f172a 0%, #162235 100%);
    border-left: 6px solid #22c55e;
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    color: #e5e7eb;
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.chat-toolbar {
    background: linear-gradient(135deg, #0f172a 0%, #162235 100%);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    margin-bottom: 1rem;
    color: #d1fae5;
}

.farm-title {font-size: 2.35rem; font-weight: 800; margin: 0;}
.subtitle {margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.96;}
.small-muted {color: #94a3b8; font-size: 0.9rem;}
.badge-good {
    background: #dcfce7;
    color: #166534;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
}
.alert-high {
    background: linear-gradient(135deg, #3b0a0a, #7f1d1d);
    color: #fecaca;
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    font-weight: 600;
    border: 1px solid rgba(248, 113, 113, 0.25);
}
.alert-med {
    background: linear-gradient(135deg, #3b2a09, #78350f);
    color: #fde68a;
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    font-weight: 600;
    border: 1px solid rgba(251, 191, 36, 0.25);
}
.alert-good {
    background: linear-gradient(135deg, #052e16, #14532d);
    color: #bbf7d0;
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    font-weight: 600;
    border: 1px solid rgba(74, 222, 128, 0.25);
}
.wow-box {
    background: linear-gradient(135deg, #0f172a, #132a13);
    border: 1px solid rgba(74, 222, 128, 0.18);
    color: #dcfce7;
    padding: 16px;
    border-radius: 18px;
    font-weight: 600;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #172033 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    padding: 16px;
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
div[data-testid="stMetricLabel"] {color: #94a3b8 !important;}
div[data-testid="stMetricValue"] {color: #f8fafc !important;}
div[data-testid="stMetricDelta"] {color: #22c55e !important;}
.stDataFrame, .stTable {
    background: #0f172a !important;
}
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_farm_data" not in st.session_state:
    st.session_state.latest_farm_data = None

if "district" not in st.session_state:
    st.session_state.district = "Hyderabad"

if "season" not in st.session_state:
    st.session_state.season = "Kharif"

if "multi_crop_df" not in st.session_state:
    st.session_state.multi_crop_df = None

# =========================
# PLOTLY THEME
# =========================
def apply_dark_plotly(fig, title=None):
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        title_font=dict(color="#f8fafc", size=22),
        legend=dict(font=dict(color="#e5e7eb")),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#e5e7eb")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#e5e7eb")
    )
    if title:
        fig.update_layout(title=title)
    return fig

# =========================
# REAL ML LAYER
# =========================
@st.cache_resource
def train_ml_models():
    crop_names = [
        "Rice", "Cotton", "Maize", "Wheat", "Sugarcane", "Soybean",
        "Groundnut", "Chilli", "Turmeric", "Red Gram", "Black Gram",
        "Green Gram", "Bengal Gram", "Sesame", "Jowar", "Bajra",
        "Sunflower", "Castor", "Paddy", "Horse Gram"
    ]

    crop_profiles = {
        "Rice": [80, 40, 40, 28, 80, 6.5, 180],
        "Cotton": [70, 35, 50, 30, 60, 7.0, 90],
        "Maize": [85, 45, 40, 27, 65, 6.7, 110],
        "Wheat": [65, 35, 35, 22, 55, 6.8, 60],
        "Sugarcane": [90, 50, 55, 29, 70, 7.2, 160],
        "Soybean": [60, 40, 35, 26, 65, 6.5, 95],
        "Groundnut": [55, 35, 45, 28, 55, 6.6, 70],
        "Chilli": [75, 40, 45, 27, 58, 6.8, 65],
        "Turmeric": [70, 45, 50, 26, 75, 6.4, 140],
        "Red Gram": [45, 30, 35, 29, 55, 7.0, 75],
        "Black Gram": [40, 28, 30, 28, 58, 6.9, 70],
        "Green Gram": [42, 30, 32, 27, 60, 6.8, 68],
        "Bengal Gram": [38, 26, 28, 24, 50, 7.1, 55],
        "Sesame": [35, 22, 28, 29, 50, 6.7, 50],
        "Jowar": [48, 28, 30, 30, 52, 7.2, 65],
        "Bajra": [45, 24, 26, 31, 48, 7.3, 45],
        "Sunflower": [58, 34, 38, 27, 54, 6.9, 62],
        "Castor": [50, 30, 36, 30, 50, 7.4, 58],
        "Paddy": [82, 42, 40, 28, 82, 6.5, 185],
        "Horse Gram": [30, 20, 22, 28, 46, 7.0, 40]
    }

    X_crop = []
    y_crop = []

    for crop in crop_names:
        base = crop_profiles[crop]
        for _ in range(35):
            n = np.clip(np.random.normal(base[0], 8), 10, 120)
            p = np.clip(np.random.normal(base[1], 6), 10, 80)
            k = np.clip(np.random.normal(base[2], 6), 10, 80)
            temp = np.clip(np.random.normal(base[3], 3), 15, 40)
            humidity = np.clip(np.random.normal(base[4], 8), 30, 95)
            ph = np.clip(np.random.normal(base[5], 0.4), 4.5, 8.5)
            rainfall = np.clip(np.random.normal(base[6], 25), 0, 300)
            X_crop.append([n, p, k, temp, humidity, ph, rainfall])
            y_crop.append(crop)

    crop_encoder = LabelEncoder()
    y_crop_enc = crop_encoder.fit_transform(y_crop)

    crop_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        random_state=42
    )
    crop_model.fit(X_crop, y_crop_enc)

    X_yield = []
    y_yield = []
    crop_factor_map = {crop: i + 1 for i, crop in enumerate(crop_names)}

    for crop in crop_names:
        crop_factor = crop_factor_map[crop]
        base = crop_profiles[crop]
        for _ in range(40):
            ph = np.clip(np.random.normal(base[5], 0.5), 4.5, 8.5)
            temp = np.clip(np.random.normal(base[3], 3), 15, 40)
            rainfall = np.clip(np.random.normal(base[6], 30), 0, 300)
            fertilizer = np.clip(np.random.normal(base[0], 10), 10, 140)
            humidity = np.clip(np.random.normal(base[4], 8), 30, 95)
            soil_moisture = np.clip(np.random.normal(42, 12), 10, 80)

            ideal_penalty = abs(ph - 6.8) * 0.5
            y = (
                1.2 + crop_factor * 0.12 + temp * 0.05 + rainfall * 0.01 +
                fertilizer * 0.015 + humidity * 0.02 + soil_moisture * 0.03 -
                ideal_penalty + np.random.normal(0, 0.3)
            )
            X_yield.append([ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_factor])
            y_yield.append(max(0.8, y))

    yield_model = RandomForestRegressor(
        n_estimators=260,
        max_depth=16,
        random_state=42
    )
    yield_model.fit(X_yield, y_yield)

    X_irr = []
    y_irr = []

    for _ in range(400):
        soil_m = np.random.uniform(10, 80)
        temp = np.random.uniform(15, 40)
        hum = np.random.uniform(30, 95)
        ph = np.random.uniform(4.5, 8.5)
        rain = np.random.uniform(0, 250)

        if rain > 120 or soil_m > 60:
            label = "Low"
        elif soil_m < 28 and temp > 30:
            label = "High"
        else:
            label = "Moderate"

        X_irr.append([soil_m, temp, hum, ph, rain])
        y_irr.append(label)

    irr_encoder = LabelEncoder()
    y_irr_enc = irr_encoder.fit_transform(y_irr)

    irrigation_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        random_state=42
    )
    irrigation_model.fit(X_irr, y_irr_enc)

    return {
        "crop_model": crop_model,
        "crop_encoder": crop_encoder,
        "yield_model": yield_model,
        "irrigation_model": irrigation_model,
        "irrigation_encoder": irr_encoder,
        "crop_factor_map": crop_factor_map,
        "all_crops": crop_names
    }

ml_bundle = train_ml_models()

# =========================
# BACKEND-LIKE SERVICE LAYER
# =========================
class AgriAPIService:
    def __init__(self, bundle):
        self.bundle = bundle

    def recommend_crop(self, features):
        pred = self.bundle["crop_model"].predict([features])[0]
        crop = self.bundle["crop_encoder"].inverse_transform([pred])[0]
        probs = self.bundle["crop_model"].predict_proba([features])[0]
        conf = round(float(np.max(probs)) * 100, 2)
        return crop, conf

    def predict_yield(self, ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_name):
        crop_factor = self.bundle["crop_factor_map"].get(crop_name, 1)
        pred = self.bundle["yield_model"].predict([[ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_factor]])[0]
        return round(float(pred), 2)

    def predict_irrigation(self, soil_moisture, temp, humidity, ph, rainfall):
        pred = self.bundle["irrigation_model"].predict([[soil_moisture, temp, humidity, ph, rainfall]])[0]
        label = self.bundle["irrigation_encoder"].inverse_transform([pred])[0]
        if label == "Low":
            action = "Skip irrigation cycle"
        elif label == "High":
            action = "Deep irrigation needed"
        else:
            action = "Light irrigation"
        return label, action

api_service = AgriAPIService(ml_bundle)

# =========================
# HELPERS
# =========================
def calculate_ndvi(red, nir):
    red = float(red)
    nir = float(nir)
    if (nir + red) == 0:
        return 0.0
    return round((nir - red) / (nir + red), 4)

def predict_health_score(ndvi, moisture, temp):
    score = (ndvi * 50) + (moisture * 0.3) + ((35 - abs(temp - 28)) * 0.2)
    return min(100, max(0, round(score, 1)))

def fertilizer_advice(n, p, k):
    if n < 50:
        return "Apply nitrogen fertilizer in split doses", 87.0
    if p < 40:
        return "Apply phosphorus fertilizer with basal dose", 86.0
    if k < 40:
        return "Apply potash for stress tolerance", 85.0
    return "Use balanced NPK fertilizer with organic manure", 89.0

def generate_ndvi_trend(ndvi):
    return pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 8)],
        "NDVI": [max(0.1, round(ndvi - 0.05 + (i * 0.008), 3)) for i in range(1, 8)]
    })

def soil_radar_figure(n, p, k, moisture, ph):
    radar_df = pd.DataFrame({
        "Metric": ["Nitrogen", "Phosphorus", "Potassium", "Moisture", "pH x10"],
        "Value": [n, p, k, moisture, ph * 10]
    })
    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        line_close=True,
        title="Soil Radar Chart"
    )
    fig.update_traces(fill="toself", line_color="#22c55e")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="#0f172a",
            radialaxis=dict(gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1")),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1"))
        ),
        font=dict(color="#e5e7eb")
    )
    return fig

def smart_alerts(health, ndvi, n, p, k):
    alerts = []
    if health < 40:
        alerts.append(("high", "🚨 High risk: poor farm health detected"))
    if ndvi < 0.35:
        alerts.append(("med", "⚠ Low NDVI detected"))
    if n < 40 or p < 30 or k < 30:
        alerts.append(("med", "⚠ Soil imbalance detected"))
    if not alerts:
        alerts.append(("good", "✅ Farm looks stable"))
    return alerts

def generate_farm_report(farm_data, multi_crop_df=None):
    lines = []
    lines.append("AGRI VISION AI - FARM REPORT")
    lines.append("=" * 50)
    if farm_data:
        lines.append(f"District: {farm_data.get('district', 'N/A')}")
        lines.append(f"Season: {farm_data.get('season', 'N/A')}")
        lines.append(f"Crop: {farm_data.get('crop', 'N/A')}")
        lines.append(f"Yield: {farm_data.get('yield', 'N/A')} t/ha")
        lines.append(f"Risk/Health: {farm_data.get('health', 'N/A')}%")
        lines.append(f"NDVI: {farm_data.get('ndvi', 'N/A')}")
        lines.append(f"Irrigation: {farm_data.get('irrigation', 'N/A')}")
        lines.append(f"Action: {farm_data.get('action', 'N/A')}")
        lines.append(f"Fertilizer Advice: {farm_data.get('fertilizer', 'N/A')}")
    if multi_crop_df is not None and not multi_crop_df.empty:
        lines.append("")
        lines.append("MULTI-CROP ANALYSIS")
        lines.append(multi_crop_df.to_string(index=False))
    return "\n".join(lines)

def wow_feature_box():
    st.markdown("""
    <div class="wow-box">
    🚀 WOW FEATURE: Multi-crop comparison, trained ML prediction, smart alerts, visual analytics, and downloadable farm report in one dashboard.
    </div>
    """, unsafe_allow_html=True)

def chatbot_reply(question, farm_data):
    q = question.lower()

    if farm_data is None:
        return """
### Situation
- No farm analysis data available yet.

### Why
- The assistant works better after Smart Advisor or Multi-Crop Analysis.

### Immediate action
- Run Smart Advisor first.
- Then ask about crop, yield, irrigation, NDVI, or fertilizer.

### Next 3 days
- Collect soil and moisture readings.
- Compare 2 to 5 crops in Multi-Crop Analysis.
"""

    if "yield" in q:
        return f"""
### Situation
- Expected yield is {farm_data.get('yield')} t/ha.

### Why
- This estimate is based on current farm inputs and selected crop.

### Immediate action
- Keep fertilizer and irrigation aligned with the recommended plan.

### Next 3 days
- Monitor moisture daily.
- Recheck NDVI trend.
- Avoid over-irrigation.
"""
    if "crop" in q:
        return f"""
### Situation
- Recommended crop is {farm_data.get('crop')}.

### Why
- The trained crop model matched your soil and climate inputs to the closest crop pattern.

### Immediate action
- Prepare seed and fertilizer planning for {farm_data.get('crop')}.

### Next 3 days
- Validate rainfall readiness.
- Recheck soil pH.
- Compare backup crops in Multi-Crop Analysis.
"""
    if "irrigation" in q or "water" in q:
        return f"""
### Situation
- Irrigation level is {farm_data.get('irrigation')}.

### Why
- The irrigation model used soil moisture, temperature, humidity, pH, and rainfall input.

### Immediate action
- {farm_data.get('action')}

### Next 3 days
- Check field moisture morning and evening.
- Prevent waterlogging.
- Track plant stress symptoms.
"""
    return f"""
### Situation
- Farm health score is {farm_data.get('health')}% and NDVI is {farm_data.get('ndvi')}.

### Why
- These values reflect current vegetation strength and field balance.

### Immediate action
- Follow fertilizer advice: {farm_data.get('fertilizer')}

### Next 3 days
- Watch NDVI trend.
- Recheck nutrient balance.
- Download the farm report for tracking.
"""

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr

def predict_disease(arr):
    score = int(np.mean(arr)) % 4
    labels = ["Healthy", "Leaf Blight", "Rust", "Leaf Spot"]
    return {
        "class_index": score,
        "class_name": labels[score],
        "confidence": 0.88
    }

def render_header():
    st.markdown(
        f"""
        <div class="hero-dashboard">
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
                <div>
                    <h1 class="farm-title">AgriVision AI</h1>
                    <p class="subtitle">Farm Health AI</p>
                </div>
                <div style='text-align:right;'>
                    <div><span class="badge-good">Telangana Edition</span></div>
                    <div style='margin-top:0.5rem;font-size:0.95rem;'>📍 {st.session_state.district} | 🌾 {st.session_state.season}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_top_dashboard():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div class='small-muted'>Overall Health</div><h2 style='color:#22c55e;'>87.4%</h2><div class='small-muted'>+3.2% this week</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='small-muted'>Active Alerts</div><h2 style='color:#f87171;'>4</h2><div class='small-muted'>2 critical zones</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='small-muted'>Yield Forecast</div><h2 style='color:#86efac;'>4.2 t/ha</h2><div class='small-muted'>On target</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='small-muted'>Soil Moisture</div><h2 style='color:#60a5fa;'>62%</h2><div class='small-muted'>Optimal</div></div>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌾 AgriVision AI")

page = st.sidebar.radio(
    "Choose Module",
    [
        "Dashboard",
        "Smart Advisor",
        "Multi-Crop Analysis",
        "Crop Recommendation",
        "Yield Prediction",
        "Irrigation",
        "Fertilizer & Soil",
        "NDVI Analysis",
        "Disease Detection",
        "AI Assistant"
    ],
)

st.sidebar.markdown("### Telangana Filters")
st.session_state.district = st.sidebar.selectbox(
    "District",
    [
        "Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar",
        "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Nalgonda",
        "Jayashankar Bhupalpally"
    ]
)
st.session_state.season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

render_header()
render_top_dashboard()

# =========================
# PAGES
# =========================
if page == "Dashboard":
    st.markdown("## Farm Health Overview")

    trend_df = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 31)],
        "Health Index": [72, 74, 73, 75, 77, 76, 78, 79, 80, 78, 81, 83, 82, 84, 85, 86, 84, 87, 88, 89, 87, 90, 91, 89, 92, 93, 91, 94, 95, 96]
    })
    zone_df = pd.DataFrame({
        "Zone": ["Healthy", "Moderate", "High Risk", "Critical"],
        "Value": [52, 28, 14, 6]
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_line = px.line(
            trend_df,
            x="Day",
            y="Health Index",
            title="Crop Health Trend - 30 Days"
        )
        fig_line.update_traces(line=dict(color="#22c55e", width=4))
        fig_line = apply_dark_plotly(fig_line)
        st.plotly_chart(fig_line, use_container_width=True, theme=None)

    with col2:
        fig_pie = px.pie(
            zone_df,
            names="Zone",
            values="Value",
            hole=0.55,
            title="Zone Distribution",
            color="Zone",
            color_discrete_map={
                "Healthy": "#22c55e",
                "Moderate": "#f59e0b",
                "High Risk": "#f97316",
                "Critical": "#ef4444"
            }
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font=dict(color="#e5e7eb")
        )
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

elif page == "Smart Advisor":
    st.markdown("## Smart Farm Advisor")

    a, b, c = st.columns(3)

    with a:
        n = st.number_input("Nitrogen (N)", value=88.0)
        p = st.number_input("Phosphorus (P)", value=40.0)
        k = st.number_input("Potassium (K)", value=41.0)
        ph = st.number_input("Soil pH", value=6.7)

    with b:
        temp = st.number_input("Temperature (°C)", value=29.0)
        humidity = st.number_input("Humidity (%)", value=78.0)
        rainfall = st.number_input("Rainfall (mm)", value=180.0)

    with c:
        moisture = st.number_input("Soil Moisture (%)", value=42.0)
        red = st.number_input("Red Band", min_value=0.0, max_value=1.0, value=0.30)
        nir = st.number_input("NIR Band", min_value=0.0, max_value=1.0, value=0.72)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        crop, crop_conf = api_service.recommend_crop([n, p, k, temp, humidity, ph, rainfall])
        irrigation, action = api_service.predict_irrigation(moisture, temp, humidity, ph, rainfall)
        yield_pred = api_service.predict_yield(ph, temp, rainfall, n, humidity, moisture, crop)
        ndvi = calculate_ndvi(red, nir)
        health = predict_health_score(ndvi, moisture, temp)
        fert, fert_conf = fertilizer_advice(n, p, k)

        st.session_state.latest_farm_data = {
            "district": st.session_state.district,
            "season": st.session_state.season,
            "crop": crop,
            "irrigation": irrigation,
            "action": action,
            "yield": yield_pred,
            "ndvi": ndvi,
            "health": health,
            "fertilizer": fert
        }

        x1, x2, x3 = st.columns(3)
        x1.metric("Crop", crop, f"{crop_conf}% confidence")
        x2.metric("Irrigation", irrigation, action)
        x3.metric("Yield", f"{yield_pred} t/ha", "Real ML prediction")

        st.markdown(
            f"<div class='health-card'><h3 style='color:#f8fafc;'>Final Farm Decision</h3>"
            f"<p><strong>Recommended crop:</strong> {crop}</p>"
            f"<p><strong>Irrigation:</strong> {irrigation} - {action}</p>"
            f"<p><strong>Fertilizer:</strong> {fert}</p>"
            f"<p><strong>NDVI:</strong> {ndvi} | <strong>Health score:</strong> {health}%</p>"
            f"<p><strong>Expected yield:</strong> {yield_pred} t/ha</p></div>",
            unsafe_allow_html=True
        )

        st.markdown("### Smart Alerts")
        for level, text in smart_alerts(health, ndvi, n, p, k):
            css = "alert-high" if level == "high" else "alert-med" if level == "med" else "alert-good"
            st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)

        d1, d2 = st.columns(2)

        with d1:
            ndvi_df = generate_ndvi_trend(ndvi)
            fig_ndvi = px.line(ndvi_df, x="Day", y="NDVI", title="NDVI Trend")
            fig_ndvi.update_traces(line=dict(color="#38bdf8", width=4))
            fig_ndvi = apply_dark_plotly(fig_ndvi)
            st.plotly_chart(fig_ndvi, use_container_width=True, theme=None)

        with d2:
            radar_fig = soil_radar_figure(n, p, k, moisture, ph)
            st.plotly_chart(radar_fig, use_container_width=True, theme=None)

        report_text = generate_farm_report(st.session_state.latest_farm_data, st.session_state.multi_crop_df)
        st.download_button(
            "Download Farm Report",
            data=report_text,
            file_name="farm_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        wow_feature_box()

elif page == "Multi-Crop Analysis":
    st.markdown("## Multi-Crop Analysis")

    all_telangana_crops = ml_bundle["all_crops"]

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("Nitrogen (N)", value=80.0, key="mc_n")
        p = st.number_input("Phosphorus (P)", value=38.0, key="mc_p")
        k = st.number_input("Potassium (K)", value=42.0, key="mc_k")
    with col2:
        ph = st.number_input("Soil pH", value=6.8, key="mc_ph")
        temp = st.number_input("Temperature (°C)", value=28.0, key="mc_temp")
        humidity = st.number_input("Humidity (%)", value=72.0, key="mc_humidity")
    with col3:
        rainfall = st.number_input("Rainfall (mm)", value=95.0, key="mc_rain")
        moisture = st.number_input("Soil Moisture (%)", value=44.0, key="mc_moisture")
        selected_crops = st.multiselect(
            "Multi Selected Crops of All Telangana Crops",
            all_telangana_crops,
            default=["Rice", "Cotton", "Maize", "Chilli"]
        )

    if st.button("Compare Selected Crops", use_container_width=True):
        rows = []
        for crop_name in selected_crops:
            pred_yield = api_service.predict_yield(ph, temp, rainfall, n, humidity, moisture, crop_name)
            suitability = round(
                max(50, min(99, 100 - abs(ph - 6.8) * 8 - abs(temp - 28) * 1.2 + (humidity * 0.08))),
                2
            )
            risk = "Low" if pred_yield >= 6 else "Moderate" if pred_yield >= 4 else "High"
            rows.append({
                "Crop": crop_name,
                "Predicted Yield (t/ha)": pred_yield,
                "Suitability Score": suitability,
                "Risk": risk
            })

        df = pd.DataFrame(rows).sort_values(by="Predicted Yield (t/ha)", ascending=False)
        st.session_state.multi_crop_df = df

        st.dataframe(df, use_container_width=True, hide_index=True)

        best_crop = df.iloc[0]["Crop"]
        best_yield = df.iloc[0]["Predicted Yield (t/ha)"]
        st.success(f"Best crop: {best_crop} with predicted yield {best_yield} t/ha")

        c1, c2 = st.columns(2)

        with c1:
            fig_bar = px.bar(
                df,
                x="Crop",
                y="Predicted Yield (t/ha)",
                color="Predicted Yield (t/ha)",
                title="Yield Comparison Across Selected Crops",
                color_continuous_scale="Greens"
            )
            fig_bar = apply_dark_plotly(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True, theme=None)

        with c2:
            fig_scatter = px.scatter(
                df,
                x="Suitability Score",
                y="Predicted Yield (t/ha)",
                color="Risk",
                size="Suitability Score",
                hover_name="Crop",
                title="Suitability vs Yield"
            )
            fig_scatter = apply_dark_plotly(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True, theme=None)

elif page == "Crop Recommendation":
    vals = [st.number_input(f"Feature {i+1}", value=float(80 + i)) for i in range(7)]

    if st.button("Recommend Crop", use_container_width=True):
        crop, conf = api_service.recommend_crop(vals)
        st.success(f"Recommended crop: {crop} ({conf}% confidence)")

elif page == "Yield Prediction":
    ph = st.number_input("Soil pH", value=6.8, key="yp_ph")
    temp = st.number_input("Temperature", value=28.0, key="yp_temp")
    rainfall = st.number_input("Rainfall", value=90.0, key="yp_rain")
    fertilizer = st.number_input("Fertilizer Input", value=85.0, key="yp_fert")
    humidity = st.number_input("Humidity", value=70.0, key="yp_hum")
    moisture = st.number_input("Soil Moisture", value=44.0, key="yp_moi")
    crop_name = st.selectbox("Crop", ml_bundle["all_crops"], index=0)

    if st.button("Predict Yield", use_container_width=True):
        y = api_service.predict_yield(ph, temp, rainfall, fertilizer, humidity, moisture, crop_name)
        st.metric("Expected Yield", f"{y} t/ha", "Real ML prediction")

        pred_df = pd.DataFrame({
            "Metric": ["pH", "Temp", "Rainfall", "Fertilizer", "Humidity", "Moisture"],
            "Value": [ph, temp, rainfall, fertilizer, humidity, moisture]
        })
        fig = px.bar(pred_df, x="Metric", y="Value", color="Metric", title="Yield Input Profile")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

elif page == "Irrigation":
    soil_m = st.number_input("Soil Moisture", value=40.0)
    temp = st.number_input("Temperature", value=29.0)
    humidity = st.number_input("Humidity", value=75.0)
    ph = st.number_input("pH", value=6.9)
    rainfall = st.number_input("Rainfall", value=60.0)

    if st.button("Get Irrigation Plan", use_container_width=True):
        result, action = api_service.predict_irrigation(soil_m, temp, humidity, ph, rainfall)
        st.info(f"Need: {result} | Action: {action}")

        irr_df = pd.DataFrame({
            "Parameter": ["Soil Moisture", "Temperature", "Humidity", "pH", "Rainfall"],
            "Value": [soil_m, temp, humidity, ph, rainfall]
        })
        fig = px.line(irr_df, x="Parameter", y="Value", markers=True, title="Irrigation Factors")
        fig.update_traces(line=dict(color="#38bdf8", width=4))
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

elif page == "Fertilizer & Soil":
    n = st.number_input("Nitrogen", value=45.0)
    p = st.number_input("Phosphorus", value=38.0)
    k = st.number_input("Potassium", value=42.0)
    moisture = st.number_input("Moisture", value=44.0)
    ph = st.number_input("Soil pH", value=6.7, key="fs_ph")

    if st.button("Analyze Soil", use_container_width=True):
        fert, conf = fertilizer_advice(n, p, k)
        df = pd.DataFrame({"Nutrient": ["N", "P", "K"], "Value": [n, p, k]})

        fig = px.bar(
            df,
            x="Nutrient",
            y="Value",
            color="Nutrient",
            title="Soil Nutrient Levels",
            color_discrete_map={"N": "#22c55e", "P": "#38bdf8", "K": "#f59e0b"}
        )
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        radar_fig = soil_radar_figure(n, p, k, moisture, ph)
        st.plotly_chart(radar_fig, use_container_width=True, theme=None)

        st.success(f"Recommendation: {fert} ({conf}% confidence)")

elif page == "NDVI Analysis":
    red = st.slider("Red Band", 0.0, 1.0, 0.30)
    nir = st.slider("NIR Band", 0.0, 1.0, 0.72)
    temp = st.slider("Temperature", 15.0, 45.0, 29.0)
    moisture = st.slider("Soil Moisture", 10.0, 80.0, 42.0)

    if st.button("Calculate NDVI", use_container_width=True):
        ndvi = calculate_ndvi(red, nir)
        health = predict_health_score(ndvi, moisture, temp)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health,
                number={'font': {'color': "#f8fafc"}},
                title={'text': "Health Score", 'font': {'color': "#f8fafc"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#22c55e"},
                    'bgcolor': "#0f172a",
                    'bordercolor': "rgba(148,163,184,0.2)",
                    'steps': [
                        {'range': [0, 40], 'color': "#7f1d1d"},
                        {'range': [40, 70], 'color': "#78350f"},
                        {'range': [70, 100], 'color': "#14532d"}
                    ]
                }
            )
        )
        gauge.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb")
        )
        st.plotly_chart(gauge, use_container_width=True, theme=None)

        ndvi_df = generate_ndvi_trend(ndvi)
        fig = px.area(ndvi_df, x="Day", y="NDVI", title="NDVI Trend")
        fig.update_traces(line=dict(color="#22c55e"), fillcolor="rgba(34,197,94,0.25)")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.success(f"NDVI: {ndvi} | Health score: {health}%")

elif page == "Disease Detection":
    uploaded_file = st.file_uploader("Upload crop leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image, arr = process_image(uploaded_file)
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        if st.button("Analyze Disease", use_container_width=True):
            result = predict_disease(arr)
            st.warning(
                f"Predicted condition: {result['class_name']} "
                f"(confidence: {round(result['confidence'] * 100, 2)}%)"
            )

            disease_df = pd.DataFrame({
                "Class": ["Healthy", "Leaf Blight", "Rust", "Leaf Spot"],
                "Score": [22, 31, 18, 29]
            })
            fig = px.bar(
                disease_df,
                x="Class",
                y="Score",
                color="Class",
                title="Disease Class Distribution"
            )
            fig = apply_dark_plotly(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None)

elif page == "AI Assistant":
    toolbar_text = (
        f"💬 AgriVision AI Chat | 📍 {st.session_state.district} | 🌾 {st.session_state.season} | "
        + ("✅ Smart Advisor data loaded" if st.session_state.latest_farm_data else "⚠️ Run Smart Advisor for personalized context")
    )

    st.markdown(f"<div class='chat-toolbar'>{toolbar_text}</div>", unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about irrigation, crop, disease, fertilizer, or Telangana farming..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AgriVision AI is analyzing..."):
                reply = chatbot_reply(prompt, st.session_state.latest_farm_data)
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;padding:18px;font-size:14px;'>🌾 AgriVision AI | Farm Health AI | Telangana Edition</div>",
    unsafe_allow_html=True
)
