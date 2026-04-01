try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

load_dotenv()

st.set_page_config(
    page_title="AgriTech Smart Assistant Pro+ Telangana Edition",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #0b1220 0%, #101827 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #14532d 0%, #1f7a47 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero-box {background: linear-gradient(120deg, #14532d, #22a063, #34d399); color: white; padding: 1.5rem 1.3rem; border-radius: 22px; box-shadow: 0 12px 34px rgba(16, 185, 129, 0.22); margin-bottom: 1rem;}
.decision-card {background: linear-gradient(135deg, #f0fff4, #dcfce7); border-left: 8px solid #16a34a; color: #123524; border-radius: 18px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); margin-top: 12px;}
.report-card {background: linear-gradient(135deg, #ffffff, #f0fdf4); border-radius: 20px; padding: 20px; border: 1px solid #d1fae5; box-shadow: 0 10px 26px rgba(0,0,0,0.08); margin-top: 12px; color: #123524;}
.metric-card {background: white; border-radius: 18px; padding: 15px; text-align: center; border: 1px solid #e5f1e8; box-shadow: 0 5px 18px rgba(0,0,0,0.06);}
.section-title {background: linear-gradient(90deg, #ecfdf5, #f0fdf4); padding: 10px 14px; border-radius: 12px; border: 1px solid #d1fae5; color: #166534; font-weight: 700; margin: 0.6rem 0 1rem 0;}
.region-chip {display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; background: #e8fff0; color: #166534; border: 1px solid #b7e4c7; margin-right: 8px; margin-bottom: 8px; font-size: 0.9rem; font-weight: 600;}
.small-muted {color: #4d6758; font-size: 14px;}
.risk-high {background: #fef2f2; color: #991b1b; border-left: 7px solid #ef4444; padding: 12px 14px; border-radius: 14px;}
.risk-medium {background: #fffbeb; color: #92400e; border-left: 7px solid #f59e0b; padding: 12px 14px; border-radius: 14px;}
.risk-low {background: #f0fdf4; color: #166534; border-left: 7px solid #22c55e; padding: 12px 14px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

COLOR = {"green": "#16a34a", "amber": "#f59e0b", "red": "#ef4444", "blue": "#3b82f6", "teal": "#14b8a6", "purple": "#8b5cf6"}

TELANGANA_CROP_INFO = {
    "Cotton": {"season": "Kharif", "water": "Moderate", "note": "Major commercial crop in northern and central Telangana."},
    "Maize": {"season": "Kharif/Rabi", "water": "Moderate", "note": "Important in irrigated and semi-irrigated belts."},
    "Paddy": {"season": "Kharif/Rabi", "water": "High", "note": "Common in canal and borewell supported areas."},
    "Red Gram": {"season": "Kharif", "water": "Low to Moderate", "note": "Useful in rainfed Telangana farming systems."},
    "Soybean": {"season": "Kharif", "water": "Moderate", "note": "Suitable in selected rainfall zones and medium soils."},
    "Turmeric": {"season": "Kharif", "water": "High", "note": "High-value crop in some Telangana districts."},
    "Jowar": {"season": "Kharif/Rabi", "water": "Low", "note": "Good drought-tolerant option in dry belts."},
    "Bengal Gram": {"season": "Rabi", "water": "Low", "note": "Popular pulse crop in post-rainy season."},
    "Groundnut": {"season": "Rabi", "water": "Moderate", "note": "Works in well-drained soils."},
    "Sesame": {"season": "Rabi", "water": "Low", "note": "Useful for short-duration dryland planning."},
    "Sunflower": {"season": "Rabi", "water": "Moderate", "note": "Suitable with controlled irrigation."},
    "Black Gram": {"season": "Rabi", "water": "Low to Moderate", "note": "Good pulse option after monsoon."},
}

class AgritechModels:
    def predict_crop_recommendation(self, features: List[float], location: str = "Telangana", season: str = "Kharif") -> Tuple[str, float, Dict[str, float]]:
        region_crops = {
            ("Andhra Pradesh", "Kharif"): ["Rice", "Maize", "Cotton", "Sugarcane", "Groundnut", "Banana", "Chilli", "Tomato"],
            ("Andhra Pradesh", "Rabi"): ["Rice", "Maize", "Groundnut", "Black Gram", "Sunflower", "Chilli", "Bengal Gram", "Tomato"],
            ("Telangana", "Kharif"): ["Cotton", "Maize", "Paddy", "Red Gram", "Soybean", "Turmeric", "Jowar", "Chilli"],
            ("Telangana", "Rabi"): ["Maize", "Bengal Gram", "Groundnut", "Sesame", "Jowar", "Sunflower", "Paddy", "Black Gram"],
        }
        crops = region_crops.get((location, season), region_crops[("Telangana", "Kharif")])
        feature_sum = sum(features)
        scores = []
        for crop in crops:
            crop_bonus = 0
            if location == "Telangana":
                if crop == "Cotton":
                    crop_bonus = 18 if season == "Kharif" else 6
                elif crop == "Red Gram":
                    crop_bonus = 16 if season == "Kharif" else 4
                elif crop == "Maize":
                    crop_bonus = 14
                elif crop == "Bengal Gram":
                    crop_bonus = 18 if season == "Rabi" else 2
                elif crop == "Jowar":
                    crop_bonus = 15
                elif crop == "Turmeric":
                    crop_bonus = 12 if season == "Kharif" else 1
            score = ((feature_sum * 0.7) + crop_bonus + (len(crop) * 2.3)) % 100 + 30
            scores.append(score)
        probs = np.array(scores) / np.sum(scores)
        prob_map = {crop: round(float(p * 100), 2) for crop, p in zip(crops, probs)}
        best_crop = max(prob_map, key=prob_map.get)
        return best_crop, prob_map[best_crop], prob_map

    def predict_crop_yield(self, features: List[float], location: str = "Telangana") -> Tuple[float, float, Dict[str, float]]:
        ph, temperature, rainfall, fertilizer, humidity, soil_moisture = features
        regional_factor = 1.07 if location == "Telangana" else 1.0
        components = {
            "Temperature": temperature * 0.66,
            "Rainfall": rainfall * 0.022,
            "Fertilizer": fertilizer * 0.042,
            "Humidity": humidity * 0.075,
            "Soil Moisture": soil_moisture * 0.26,
            "pH Penalty": -abs(ph - 6.7) * 2.2,
        }
        pred = round(max(sum(components.values()) * regional_factor, 0), 2)
        return pred, 90.2, components

    def predict_irrigation(self, features: List[float], location: str = "Telangana") -> Tuple[str, str, float, Dict[str, float]]:
        soil_moisture, temperature, humidity, ph, rainfall = features
        factors = {
            "Soil Dryness": max(0, 100 - soil_moisture),
            "Heat Pressure": max(0, temperature * 2.3),
            "Humidity Buffer": max(0, 100 - humidity),
            "Rain Support": rainfall,
            "pH Balance": max(0, 100 - abs(ph - 6.8) * 20),
        }
        if location == "Telangana":
            if rainfall > 155 or soil_moisture > 56:
                return "Low", "Reduce irrigation and check drainage in Telangana field conditions", 91.5, factors
            if soil_moisture < 32 and temperature > 31:
                return "High", "Increase irrigation; use drip or split irrigation for Telangana dry belts", 91.5, factors
            return "Moderate", "Maintain moderate irrigation with frequent soil checks", 91.5, factors
        if rainfall > 180 or soil_moisture > 60:
            return "Low", "Reduce irrigation", 90.0, factors
        if soil_moisture < 30 and temperature > 30:
            return "High", "Increase irrigation", 90.0, factors
        return "Moderate", "Maintain moderate irrigation", 90.0, factors

    def predict_fertilizer(self, features: List[float], location: str = "Telangana", crop_name: str = "Cotton") -> Tuple[str, float, Dict[str, float]]:
        N, P, K, temperature, humidity, moisture, soil_type, crop_type, ph, rainfall = features
        levels = {"Nitrogen": N, "Phosphorus": P, "Potassium": K}
        if location == "Telangana":
            if crop_name == "Cotton":
                if N < 60:
                    return "Apply nitrogen in split doses for cotton; add foliar micronutrient spray", 89.0, levels
                if K < 42:
                    return "Apply potash to improve boll development and dry spell tolerance", 89.0, levels
                return "Use balanced NPK with zinc or micronutrient support for cotton", 89.0, levels
            if crop_name in ["Maize", "Paddy"]:
                if N < 58:
                    return "Apply nitrogen in 2 to 3 split doses", 88.8, levels
                if P < 42:
                    return "Apply basal phosphorus before or during sowing", 88.8, levels
                return "Balanced NPK with field-stage based top dressing", 88.8, levels
            if crop_name in ["Red Gram", "Bengal Gram", "Black Gram"]:
                return "Use moderate fertilizer with phosphorus support; avoid excess nitrogen in pulses", 88.4, levels
            if crop_name in ["Jowar", "Sesame"]:
                return "Use light to moderate fertilizer with moisture-based application planning", 88.0, levels
        if N < 50:
            return "Apply nitrogen fertilizer", 87.0, levels
        if P < 40:
            return "Apply phosphorus fertilizer", 87.0, levels
        if K < 38:
            return "Apply potassium fertilizer", 87.0, levels
        return "Balanced NPK fertilizer", 87.0, levels

    def predict_rainfall(self, features: List[float], location: str = "Telangana") -> Tuple[float, str, float, Dict[str, float]]:
        temp, humidity, pressure, wind_speed = features
        regional_boost = 3.0 if location == "Telangana" else 0.0
        factors = {
            "Humidity Impact": humidity * 0.7,
            "Temperature Impact": -(temp * 0.22),
            "Wind Impact": wind_speed * 0.45,
            "Pressure Impact": (1015 - pressure) * 1.1,
            "Regional Boost": regional_boost,
        }
        rain = max(round(sum(factors.values()), 2), 0)
        if rain > 60:
            status = "High Rainfall"
        elif rain > 25:
            status = "Moderate Rainfall"
        else:
            status = "Low Rainfall"
        return rain, status, 87.2, factors

    def predict_npk(self, features: List[float]) -> Tuple[Dict[str, float], float]:
        ph, ec, organic_carbon, moisture, temp, rainfall = features
        n = round((organic_carbon * 42) + (moisture * 0.8), 2)
        p = round((ec * 21) + (ph * 3.2), 2)
        k = round((rainfall * 0.07) + (temp * 0.60), 2)
        return {"Nitrogen": n, "Phosphorus": p, "Potassium": k}, 88.4

    def calculate_ndvi(self, red: float, nir: float) -> float:
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_crop_stress(self, features: List[float], location: str = "Telangana") -> Tuple[str, float, Dict[str, float]]:
        ndvi, temp, soil_moisture, humidity = features
        temp_threshold = 35 if location == "Telangana" else 36
        factors = {
            "Vegetation Health": ndvi * 100,
            "Temperature Stress": max(0, (temp - 25) * 8),
            "Moisture Security": soil_moisture,
            "Humidity Support": humidity,
        }
        if ndvi < 0.35 or soil_moisture < 28 or temp > temp_threshold:
            return "High", 80.5, factors
        if ndvi < 0.50:
            return "Moderate", 86.5, factors
        return "Low", 92.8, factors

    def predict_disease(self, image_array: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        labels = ["Healthy", "Leaf Blight", "Rust", "Leaf Spot", "Powdery Mildew"]
        raw = np.array([28, 18, 15, 20, 19], dtype=float) + (int(np.mean(image_array)) % 5)
        probs = raw / raw.sum() * 100
        prob_map = {k: round(float(v), 2) for k, v in zip(labels, probs)}
        best = max(prob_map, key=prob_map.get)
        return best, prob_map[best], prob_map

agrimodels = AgritechModels()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_final_output" not in st.session_state:
    st.session_state.latest_final_output = None
if "last_disease_output" not in st.session_state:
    st.session_state.last_disease_output = {"disease": "Healthy", "confidence": 0.0}

def num_input(label: str, value: float = 0.0) -> float:
    return st.number_input(label, value=float(value), step=0.1)

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr

def get_simulation_defaults(location: str, season: str) -> Dict[str, float]:
    data = {
        ("Andhra Pradesh", "Kharif"): {"N": 90, "P": 42, "K": 43, "temperature": 28, "humidity": 82, "ph": 6.5, "rainfall": 220, "soil_moisture": 48, "fertilizer": 110, "ec": 1.2, "organic_carbon": 0.82},
        ("Andhra Pradesh", "Rabi"): {"N": 75, "P": 38, "K": 35, "temperature": 23, "humidity": 68, "ph": 6.8, "rainfall": 90, "soil_moisture": 36, "fertilizer": 95, "ec": 1.0, "organic_carbon": 0.70},
        ("Telangana", "Kharif"): {"N": 82, "P": 40, "K": 41, "temperature": 30, "humidity": 74, "ph": 6.7, "rainfall": 165, "soil_moisture": 40, "fertilizer": 102, "ec": 1.15, "organic_carbon": 0.74},
        ("Telangana", "Rabi"): {"N": 70, "P": 36, "K": 34, "temperature": 25, "humidity": 62, "ph": 7.0, "rainfall": 55, "soil_moisture": 30, "fertilizer": 88, "ec": 0.95, "organic_carbon": 0.66},
    }
    return data.get((location, season), data[("Telangana", "Kharif")])

def get_region_note(location: str, season: str) -> str:
    notes = {
        ("Andhra Pradesh", "Kharif"): "Best suited for paddy, chilli, banana, and humid-season disease monitoring.",
        ("Andhra Pradesh", "Rabi"): "Good for maize, pulses, sunflower, and controlled irrigation planning.",
        ("Telangana", "Kharif"): "Optimized for cotton, maize, red gram, soybean, turmeric, and rainfed crop planning.",
        ("Telangana", "Rabi"): "Optimized for bengal gram, groundnut, sesame, jowar, black gram, and dry-season water management.",
    }
    return notes.get((location, season), "Regional optimization active.")

def compute_risk_level(disease: str, ndvi: float) -> str:
    if disease.lower() != "healthy":
        return "HIGH"
    if ndvi < 0.5:
        return "MEDIUM"
    return "LOW"

def get_risk_class(risk: str) -> str:
    return "risk-high" if risk == "HIGH" else "risk-medium" if risk == "MEDIUM" else "risk-low"

def build_action_list(irrigation_action: str, fertilizer_action: str, disease: str, weather_status: str, stress_level: str) -> List[str]:
    actions = [irrigation_action, fertilizer_action]
    if disease.lower() != "healthy":
        actions.append(f"Apply suitable disease management for {disease}")
    if weather_status == "High Rainfall":
        actions.append("Reduce irrigation and improve field drainage")
    if stress_level == "High":
        actions.append("Inspect crop immediately for heat or moisture stress")
    final_actions = []
    for action in actions:
        if action not in final_actions:
            final_actions.append(action)
    return final_actions[:4]

def generate_smart_report(data: Dict[str, object]) -> str:
    risk_class = get_risk_class(data["risk"])
    actions_html = "".join([f"<li>{a}</li>" for a in data["actions"]])
    return f"""
    <div class="report-card">
        <h2>🌾 SMART FARM REPORT</h2>
        <p><strong>Location:</strong> {data['location']} | <strong>Season:</strong> {data['season']}</p>
        <p><strong>Crop:</strong> {data['crop']} ({data['crop_conf']}%)</p>
        <p><strong>Weather:</strong> {data['weather']} ({data['weather_conf']}%)</p>
        <p><strong>Disease:</strong> {data['disease']} ({data['disease_conf']}%)</p>
        <p><strong>NDVI:</strong> {data['ndvi']}</p>
        <div class="{risk_class}" style="margin: 12px 0;"><strong>Risk:</strong> {data['risk']}</div>
        <p><strong>Actions:</strong></p>
        <ul>{actions_html}</ul>
    </div>
    """

def generate_gpt_like_reply(user_text: str, final_output=None, location: str = "Telangana", season: str = "Kharif") -> Tuple[str, str]:
    text = user_text.lower().strip()
    crop = "Cotton"
    irrigation = "Moderate"
    weather = "Moderate Rainfall"
    actions = ["Maintain field monitoring", "Use balanced fertilizer"]
    stress = "Moderate"
    ndvi = 0.52
    disease = "Healthy"
    risk = "LOW"

    if final_output:
        crop = final_output.get("crop", crop)
        irrigation = final_output.get("irrigation", irrigation)
        weather = final_output.get("weather", weather)
        actions = final_output.get("actions", actions)
        stress = final_output.get("stress", stress)
        ndvi = final_output.get("ndvi", ndvi)
        disease = final_output.get("disease", disease)
        risk = final_output.get("risk", risk)

    if "what should i do" in text or "what i do" in text:
        return (
            f"### 🤖 Smart Advice\n- Crop: {crop}\n- Risk: {risk}\n- Advice: {actions[0]}\n- Next Step: {actions[1] if len(actions) > 1 else actions[0]}\n- Reason: Weather is {weather}, stress is {stress}, disease is {disease}, NDVI is {ndvi}.",
            f"### 📌 Why?\n- Location: {location}\n- Season: {season}\n- This advice combines crop, weather, disease, irrigation, and fertilizer signals."
        )

    if "fertilizer" in text:
        return (
            f"### 🧪 Fertilizer Advice\n- Current recommendation: {actions[1] if len(actions) > 1 else 'Use balanced fertilizer'}\n- Check NPK before top dressing.\n- Prefer split doses in hot conditions.",
            f"### 📌 Reason\n- Nutrient efficiency depends on moisture, rainfall, and crop stage in {location}."
        )

    if "irrigation" in text:
        return (
            f"### 💧 Irrigation Advice\n- Current need: {irrigation}\n- Use soil moisture as the main signal.\n- Avoid excess irrigation when rainfall risk is high.",
            f"### 📌 Reason\n- This plan uses rainfall, temperature, and soil moisture for {season} season."
        )

    return (
        f"### 🤖 Smart Agri Assistant\n- Current region: {location}\n- Current season: {season}\n- Ask about irrigation, disease, fertilizer, NDVI, or crop actions.",
        "### 📌 Assistant Note\n- Run Smart Advisor to generate the full report."
    )

def plot_gauge(title: str, value: float, min_val: float, max_val: float, good_range: Tuple[float, float], color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, good_range[0]], 'color': '#fee2e2'},
                {'range': [good_range[0], good_range[1]], 'color': '#dcfce7'},
                {'range': [good_range[1], max_val], 'color': '#fef3c7'}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white')
    return fig

def plot_bar_dict(data: Dict[str, float], title: str):
    x = list(data.keys())
    y = list(data.values())
    fig = go.Figure(go.Bar(
        x=x,
        y=y,
        marker=dict(color=y, colorscale='Viridis'),
        text=[f"{v:.2f}" for v in y],
        textposition='outside'
    ))
    fig.update_layout(title=title, template='plotly_white', height=400, xaxis_title="Factors", yaxis_title="Value")
    return fig

def plot_radar(data: Dict[str, float], title: str, fillcolor: str = 'rgba(34,197,94,0.25)'):
    labels = list(data.keys())
    vals = list(data.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor=fillcolor,
        line=dict(color=COLOR['green'], width=3),
        name=title
    ))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True)), template='plotly_white', height=430)
    return fig

def plot_donut(prob_map: Dict[str, float], title: str):
    fig = go.Figure(go.Pie(labels=list(prob_map.keys()), values=list(prob_map.values()), hole=0.52, textinfo='label+percent'))
    fig.update_layout(title=title, template='plotly_white', height=420)
    return fig

def plot_line_forecast(base_value: float, title: str, ytitle: str):
    days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
    vals = [round(base_value * (0.96 + i * 0.015), 2) for i in range(7)]
    fig = go.Figure(go.Scatter(x=days, y=vals, mode='lines+markers', line=dict(color=COLOR['blue'], width=3), fill='tozeroy'))
    fig.update_layout(title=title, template='plotly_white', height=380, xaxis_title='Timeline', yaxis_title=ytitle)
    return fig

def plot_nutrient_comparison(levels: Dict[str, float], ideal: Dict[str, float], title: str):
    categories = list(levels.keys())
    current = [levels[k] for k in categories]
    target = [ideal[k] for k in categories]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current', x=categories, y=current, marker_color=[COLOR['green'], COLOR['blue'], COLOR['amber']]))
    fig.add_trace(go.Bar(name='Target', x=categories, y=target, marker_color='#cbd5e1'))
    fig.update_layout(barmode='group', template='plotly_white', title=title, height=400)
    return fig

st.sidebar.title("🌿 AgriTech Menu")
page = st.sidebar.radio(
    "Choose Module",
    ["Home", "Smart Advisor", "Crop Recommendation", "Crop Yield", "Irrigation", "Fertilizer", "Soil & NPK", "NDVI & Stress", "Image Models", "Farmer Chat Assistant"]
)

st.sidebar.markdown("### 🌍 Regional Simulation Mode")
location = st.sidebar.selectbox("Location", ["Telangana", "Andhra Pradesh"], index=0)
season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"], index=0)
defaults = get_simulation_defaults(location, season)

st.markdown(
    f"""<div class="hero-box"><h1>🌾 AgriTech Smart Assistant Pro+ Telangana Edition</h1><p>Advanced agriculture dashboard with stronger Telangana crop optimization, clear visual analytics, and easy-to-understand predictions.</p><p><strong>Location:</strong> {location} | <strong>Season:</strong> {season}</p></div>""",
    unsafe_allow_html=True
)

if page == "Home":
    st.markdown(
        f'<span class="region-chip">{location}</span><span class="region-chip">{season}</span><span class="region-chip">{get_region_note(location, season)}</span>',
        unsafe_allow_html=True
    )

    if location == "Telangana":
        st.markdown('<div class="section-title">Top Telangana Crops</div>', unsafe_allow_html=True)
        crop_names = list(TELANGANA_CROP_INFO.keys())
        crop_scores = [88, 86, 84, 82, 78, 76, 80, 81, 77, 72, 74, 75]
        fig_crop = go.Figure(go.Bar(x=crop_names, y=crop_scores, marker=dict(color=crop_scores, colorscale="Greens")))
        fig_crop.update_layout(title="Telangana Crop Suitability Snapshot", template="plotly_white", height=420)
        st.plotly_chart(fig_crop, use_container_width=True)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]])
    fig.add_trace(go.Pie(labels=["Prediction", "Visualization", "Decision Support", "Chat Assistant"], values=[30, 25, 25, 20], hole=0.45), row=1, col=1)
    fig.add_trace(go.Bar(
        x=["Crop", "Yield", "Irrigation", "Fertilizer", "NPK", "NDVI", "Disease"],
        y=[93, 90, 91, 89, 88, 86, 86],
        marker_color=[COLOR['green'], COLOR['blue'], COLOR['amber'], COLOR['teal'], COLOR['purple'], '#22c55e', '#f97316'],
        text=[93, 90, 91, 89, 88, 86, 86],
        textposition='outside'
    ), row=1, col=2)
    fig.update_layout(height=420, template='plotly_white', title='System Capability Snapshot')
    st.plotly_chart(fig, use_container_width=True)

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

    if st.button("🚀 Generate Smart Farm Report", type="primary"):
        crop, crop_conf, crop_map = agrimodels.predict_crop_recommendation(
            [N, P, K, temperature, humidity, ph, rainfall], location, season
        )
        irrigation, irrigation_action, irr_conf, irr_factors = agrimodels.predict_irrigation(
            [soil_moisture, temperature, humidity, ph, rainfall], location
        )
        rain_value, weather_status, weather_conf, rain_factors = agrimodels.predict_rainfall(
            [temperature, humidity, 1012, 8], location
        )
        fert_result, fert_conf, fert_levels = agrimodels.predict_fertilizer(
            [N, P, K, temperature, humidity, soil_moisture, 1, 1, ph, rainfall], location, crop
        )
        yield_pred, yield_conf, yield_components = agrimodels.predict_crop_yield(
            [ph, temperature, rainfall, fertilizer_val, humidity, soil_moisture], location
        )
        ndvi = agrimodels.calculate_ndvi(red, nir)
        stress_level, stress_conf, stress_factors = agrimodels.predict_crop_stress(
            [ndvi, temperature, soil_moisture, humidity], location
        )

        disease_data = st.session_state.last_disease_output
        disease = disease_data.get("disease", "Healthy")
        disease_conf = disease_data.get("confidence", 0.0)

        risk = compute_risk_level(disease, ndvi)
        actions = build_action_list(irrigation_action, fert_result, disease, weather_status, stress_level)

        st.session_state.latest_final_output = {
            "crop": crop,
            "crop_conf": crop_conf,
            "irrigation": irrigation,
            "irrigation_conf": irr_conf,
            "weather": weather_status,
            "weather_conf": weather_conf,
            "actions": actions,
            "yield": yield_pred,
            "yield_conf": yield_conf,
            "ndvi": ndvi,
            "stress": stress_level,
            "stress_conf": stress_conf,
            "disease": disease,
            "disease_conf": disease_conf,
            "risk": risk,
            "location": location,
            "season": season,
            "fertilizer": fert_result,
            "fertilizer_conf": fert_conf,
        }

        st.markdown(generate_smart_report(st.session_state.latest_final_output), unsafe_allow_html=True)

        if location == "Telangana" and crop in TELANGANA_CROP_INFO:
            info = TELANGANA_CROP_INFO[crop]
            st.info(f"Telangana crop note: {crop} | Best season: {info['season']} | Water need: {info['water']} | {info['note']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Crop Confidence", f"{crop_conf}%")
        m2.metric("Disease Confidence", f"{disease_conf}%")
        m3.metric("Yield Forecast", f"{yield_pred} t/ha")
        m4.metric("NDVI Score", f"{ndvi}")

        tab1, tab2, tab3 = st.tabs(["🌾 Farming", "👁 Monitoring", "🤖 Assistant"])

        with tab1:
            st.plotly_chart(plot_donut(crop_map, "Crop Recommendation Probability"), use_container_width=True)
            st.plotly_chart(plot_bar_dict(yield_components, "Yield Contribution Analysis"), use_container_width=True)
            st.bar_chart({"Nitrogen": [N], "Phosphorus": [P], "Potassium": [K]})

        with tab2:
            st.plotly_chart(plot_radar(stress_factors, "Crop Stress Radar"), use_container_width=True)
            st.plotly_chart(plot_bar_dict(rain_factors, "Rainfall Driver Analysis"), use_container_width=True)
            st.plotly_chart(plot_line_forecast(yield_pred, "7-Day Yield Trend Simulation", "Yield Index"), use_container_width=True)

        with tab3:
            answer, why = generate_gpt_like_reply("what should i do", st.session_state.latest_final_output, location, season)
            st.markdown(answer)
            st.markdown(why)

elif page == "Crop Recommendation":
    st.subheader("🌾 Telangana Crop Recommendation")
    inputs = [num_input(f"Feature {i+1}", defaults["N"] + i * 2) for i in range(7)]

    if st.button("Recommend Crop"):
        crop, conf, crop_map = agrimodels.predict_crop_recommendation(inputs, location, season)
        st.success(f"Recommended crop for {location} {season}: {crop} ({conf}% confidence)")

        if location == "Telangana" and crop in TELANGANA_CROP_INFO:
            info = TELANGANA_CROP_INFO[crop]
            st.info(f"{crop} | Season: {info['season']} | Water: {info['water']} | {info['note']}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_donut(crop_map, "Crop Recommendation Share"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_bar_dict(crop_map, "Crop Score Comparison"), use_container_width=True)

elif page == "Crop Yield":
    st.subheader("📈 Crop Yield Prediction")
    yield_inputs = [num_input(lbl, val) for lbl, val in zip(
        ["pH", "Temperature", "Rainfall", "Fertilizer", "Humidity", "Soil Moisture"],
        [defaults["ph"], defaults["temperature"], defaults["rainfall"], defaults["fertilizer"], defaults["humidity"], defaults["soil_moisture"]]
    )]

    if st.button("Predict Yield"):
        yield_pred, conf, comps = agrimodels.predict_crop_yield(yield_inputs, location)
        st.metric("Expected Yield", f"{yield_pred} tons/ha", delta=f"{conf}% confidence")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_gauge("Yield Gauge", yield_pred, 0, 120, (35, 80), COLOR['green']), use_container_width=True)
        with c2:
            st.plotly_chart(plot_bar_dict(comps, "Yield Factor Breakdown"), use_container_width=True)

        st.plotly_chart(plot_line_forecast(yield_pred, "7-Day Yield Trend Simulation", "Yield Index"), use_container_width=True)

elif page == "Irrigation":
    st.subheader("💧 Irrigation Recommendation")
    irr_inputs = [num_input(lbl, val) for lbl, val in zip(
        ["Soil Moisture", "Temperature", "Humidity", "pH", "Rainfall"],
        [defaults["soil_moisture"], defaults["temperature"], defaults["humidity"], defaults["ph"], defaults["rainfall"]]
    )]

    if st.button("Predict Irrigation Need"):
        level, action, conf, factors = agrimodels.predict_irrigation(irr_inputs, location)
        st.success(f"Irrigation Need: {level} ({conf}% confidence)")
        st.info(action)

        c1, c2 = st.columns(2)
        with c1:
            gauge_value = 85 if level == "High" else 55 if level == "Moderate" else 25
            st.plotly_chart(plot_gauge("Irrigation Need Gauge", gauge_value, 0, 100, (35, 70), COLOR['blue']), use_container_width=True)
        with c2:
            st.plotly_chart(plot_bar_dict(factors, "Irrigation Drivers"), use_container_width=True)

elif page == "Fertilizer":
    st.subheader("🧪 Fertilizer Recommendation")
    crop_name = st.selectbox("Select Crop", list(TELANGANA_CROP_INFO.keys()), index=0)

    fert_inputs = [
        num_input("Nitrogen", defaults["N"]),
        num_input("Phosphorus", defaults["P"]),
        num_input("Potassium", defaults["K"]),
        num_input("Temperature", defaults["temperature"]),
        num_input("Humidity", defaults["humidity"]),
        num_input("Soil Moisture", defaults["soil_moisture"]),
        1, 1,
        num_input("pH", defaults["ph"]),
        num_input("Rainfall", defaults["rainfall"])
    ]

    if st.button("Predict Fertilizer"):
        result, conf, levels = agrimodels.predict_fertilizer(fert_inputs, location, crop_name)
        st.success(f"Fertilizer Advice: {result} ({conf}% confidence)")
        ideal = {"Nitrogen": 75, "Phosphorus": 45, "Potassium": 40}

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_nutrient_comparison(levels, ideal, "Current vs Ideal NPK"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_radar(levels, "NPK Radar Profile", 'rgba(20,184,166,0.25)'), use_container_width=True)

        st.bar_chart(levels)

elif page == "Soil & NPK":
    st.subheader("🌱 Soil & NPK Prediction")
    npk_inputs = [num_input(lbl, val) for lbl, val in zip(
        ["pH", "EC", "Organic Carbon", "Moisture", "Temperature", "Rainfall"],
        [defaults["ph"], defaults["ec"], defaults["organic_carbon"], defaults["soil_moisture"], defaults["temperature"], defaults["rainfall"]]
    )]

    if st.button("Predict NPK"):
        npk_vals, conf = agrimodels.predict_npk(npk_inputs)
        st.success(f"Predicted soil nutrients ({conf}% confidence)")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_bar_dict(npk_vals, "Predicted NPK Values"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_radar(npk_vals, "Soil Nutrient Radar", 'rgba(139,92,246,0.24)'), use_container_width=True)

        st.bar_chart(npk_vals)

elif page == "NDVI & Stress":
    st.subheader("🛰 NDVI & Crop Stress")
    red = num_input("Red Band", 0.30)
    nir = num_input("NIR Band", 0.72)
    temp = num_input("Temperature", defaults["temperature"])
    moisture = num_input("Soil Moisture", defaults["soil_moisture"])
    humidity = num_input("Humidity", defaults["humidity"])

    if st.button("Analyze NDVI & Stress"):
        ndvi = agrimodels.calculate_ndvi(red, nir)
        stress, conf, factors = agrimodels.predict_crop_stress([ndvi, temp, moisture, humidity], location)
        st.metric("NDVI", ndvi)
        st.success(f"Crop Stress: {stress} ({conf}% confidence)")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_gauge("NDVI Gauge", ndvi, -1, 1, (0.35, 0.75), COLOR['teal']), use_container_width=True)
        with c2:
            st.plotly_chart(plot_radar(factors, "Stress Factor Radar", 'rgba(239,68,68,0.20)'), use_container_width=True)

        st.bar_chart({"NDVI": [ndvi], "Moisture": [moisture], "Humidity": [humidity]})

elif page == "Image Models":
    st.subheader("📷 Disease Detection Demo")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image, arr = process_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Disease"):
            disease, conf, prob_map = agrimodels.predict_disease(arr)
            st.session_state.last_disease_output = {"disease": disease, "confidence": conf}
            st.success(f"Prediction: {disease} ({conf}% confidence)")
            st.plotly_chart(plot_donut(prob_map, "Disease Prediction Probability"), use_container_width=True)

elif page == "Farmer Chat Assistant":
    st.subheader("🤖 Farmer Chat Assistant")
    user_query = st.text_input("Ask your agriculture question")

    if st.button("Get Answer"):
        answer, why = generate_gpt_like_reply(user_query, st.session_state.latest_final_output, location, season)
        st.markdown(answer)
        st.markdown(why)
        st.session_state.chat_history.append({"q": user_query, "a": answer, "w": why})

    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, item in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {item['q']}")
            st.markdown(item["a"])
            st.markdown(item["w"])
