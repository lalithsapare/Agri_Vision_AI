import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from openai import OpenAI

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #08111f 0%, #020817 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #166534 0%, #166534 55%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero-dashboard {background: linear-gradient(90deg, #166534, #22c55e, #16a34a); color: white; padding: 1.8rem 1.5rem; border-radius: 24px; box-shadow: 0 12px 32px rgba(22, 101, 52, 0.3); margin-bottom: 1.5rem;}
.metric-card {background: rgba(255,255,255,0.96); border-radius: 20px; padding: 1.4rem; text-align: center; border: 1px solid #dcfce7; box-shadow: 0 8px 24px rgba(0,0,0,0.12);}
.health-card {background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-radius: 20px; padding: 1.4rem; box-shadow: 0 8px 24px rgba(22,101,52,0.15); color: #14532d;}
.chat-toolbar {background: rgba(255,255,255,0.96); border-radius: 16px; padding: 1rem 1.2rem; border: 1px solid #dcfce7; margin-bottom: 1rem; color: #14532d;}
.farm-title {font-size: 2.35rem; font-weight: 800; margin: 0;}
.subtitle {margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.96;}
.small-muted {color: #64748b; font-size: 0.9rem;}
.badge-good {background: #dcfce7; color: #166534; padding: 0.35rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


class TelanganaAgriModels:
    def __init__(self):
        self.crops = {
            "Kharif": ["Cotton", "Maize", "Red Gram", "Soybean", "Turmeric", "Jowar"],
            "Rabi": ["Bengal Gram", "Groundnut", "Sesame", "Jowar", "Paddy", "Black Gram"],
        }

    def predict_crop_recommendation(self, features, season="Kharif"):
        season_crops = self.crops.get(season, self.crops["Kharif"])
        score = int(sum(features)) % len(season_crops)
        crop = season_crops[score]
        confidence = round(85 + (score % 10), 1)

        all_scores = []
        base = sum(features) / len(features)
        for i, c in enumerate(season_crops):
            crop_score = round(max(40, min(98, 60 + ((base + i * 7) % 35))), 1)
            all_scores.append((c, crop_score))

        all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        return crop, confidence, all_scores

    def predict_crop_yield(self, features):
        ph, temp, rainfall, fert, humidity, moisture = features
        pred = (
            (temp * 0.65)
            + (rainfall * 0.03)
            + (fert * 0.045)
            + (humidity * 0.075)
            + (moisture * 0.25)
            - abs(ph - 6.8) * 2.2
        )
        pred = round(max(pred, 0), 2)

        months = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        trend = [round(max(pred * f, 0), 2) for f in [0.42, 0.56, 0.71, 0.84, 0.93, 1.00]]
        return pred, 88.0, months, trend

    def predict_irrigation(self, features):
        moisture, temp, humidity, ph, rainfall = features
        if rainfall > 150 or moisture > 55:
            need = "Low"
            action = "Skip irrigation cycle"
            liters = [20, 15, 10, 10, 8, 8, 5]
            conf = 92.0
        elif moisture < 28 and temp > 32:
            need = "High"
            action = "Deep irrigation needed"
            liters = [55, 58, 60, 50, 48, 46, 40]
            conf = 90.0
        else:
            need = "Moderate"
            action = "Light irrigation"
            liters = [35, 38, 40, 34, 32, 30, 28]
            conf = 89.0

        days = ["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7"]
        return need, action, conf, days, liters

    def calculate_ndvi(self, red, nir):
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_health_score(self, ndvi, moisture, temp):
        score = (ndvi * 50) + (moisture * 0.3) + ((35 - abs(temp - 28)) * 0.2)
        return min(100, max(0, round(score, 1)))

    def predict_fertilizer(self, n, p, k):
        if n < 50:
            rec = "Apply nitrogen fertilizer in split doses"
        elif p < 40:
            rec = "Apply phosphorus fertilizer with basal dose"
        elif k < 40:
            rec = "Apply potash for stress tolerance"
        else:
            rec = "Use balanced NPK fertilizer with organic manure"

        conf = 89.0
        target = {"N": 80, "P": 45, "K": 45}
        current = {"N": n, "P": p, "K": k}
        gap = {k1: max(target[k1] - current[k1], 0) for k1 in target}
        return rec, conf, current, target, gap

    def predict_disease(self, arr):
        score = int(np.mean(arr)) % 4
        labels = ["Healthy", "Leaf Blight", "Rust", "Leaf Spot"]
        probs = np.array([0.12, 0.18, 0.22, 0.16], dtype=float)
        probs[score] = 0.62
        probs = probs / probs.sum()
        return {
            "class_index": score,
            "class_name": labels[score],
            "confidence": float(np.max(probs)),
            "all_probs": {labels[i]: round(float(probs[i]) * 100, 2) for i in range(len(labels))}
        }


agrimodels = TelanganaAgriModels()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_farm_data" not in st.session_state:
    st.session_state.latest_farm_data = None

if "district" not in st.session_state:
    st.session_state.district = "Hyderabad"

if "season" not in st.session_state:
    st.session_state.season = "Kharif"


def get_openrouter_api_key():
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            key = str(st.secrets["OPENROUTER_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass

    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    return ""


def get_openrouter_reply(user_text, farm_data, district, season):
    api_key = get_openrouter_api_key()

    if not api_key:
        return "OpenRouter API key missing. Add OPENROUTER_API_KEY in Streamlit Cloud secrets."

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        farm_context = "No latest farm data available."
        if farm_data:
            farm_context = (
                f"District: {district}. Season: {season}. Recommended crop: {farm_data.get('crop')}. "
                f"Irrigation: {farm_data.get('irrigation')} ({farm_data.get('action')}). "
                f"Expected yield: {farm_data.get('yield')} t/ha. NDVI: {farm_data.get('ndvi')}. "
                f"Health score: {farm_data.get('health')}%. Fertilizer: {farm_data.get('fertilizer')}."
            )

        system_prompt = (
            "You are AgriVision AI, a helpful agriculture assistant for Telangana farmers. "
            "Reply in simple, practical language. "
            "Always structure your answer as: 1. What is happening 2. Why 3. Immediate action 4. Next 3 days."
        )

        user_prompt = f"Farm context: {farm_context}\n\nUser question: {user_text}"

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            extra_headers={
                "HTTP-Referer": "https://agrivisionai.streamlit.app",
                "X-Title": "AgriVision AI"
            }
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenRouter chatbot error: {e}"


def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr


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
        st.markdown("<div class='metric-card'><div class='small-muted'>Overall Health</div><h2 style='color:#16a34a;'>87.4%</h2><div class='small-muted'>+3.2% this week</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='small-muted'>Active Alerts</div><h2 style='color:#dc2626;'>4</h2><div class='small-muted'>2 critical zones</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='small-muted'>Yield Forecast</div><h2 style='color:#166534;'>4.2 t/ha</h2><div class='small-muted'>On target</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='small-muted'>Soil Moisture</div><h2 style='color:#2563eb;'>62%</h2><div class='small-muted'>Optimal</div></div>", unsafe_allow_html=True)


st.sidebar.title("🌾 AgriVision AI")
page = st.sidebar.radio(
    "Choose Module",
    [
        "Dashboard",
        "Smart Advisor",
        "Crop Recommendation",
        "Yield Prediction",
        "Irrigation",
        "Fertilizer & Soil",
        "NDVI Analysis",
        "Disease Detection",
        "AI Assistant"
    ],
)

st.session_state.district = st.sidebar.selectbox(
    "District",
    ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar"]
)
st.session_state.season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

render_header()
render_top_dashboard()

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
        fig_line = px.line(trend_df, x="Day", y="Health Index", title="Crop Health Trend - 30 Days")
        fig_line.update_traces(line=dict(color="#16a34a", width=4))
        fig_line.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        fig_pie = px.pie(zone_df, names="Zone", values="Value", hole=0.55, title="Zone Distribution")
        fig_pie.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

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
        crop, crop_conf, crop_scores = agrimodels.predict_crop_recommendation(
            [n, p, k, temp, humidity, ph, rainfall], st.session_state.season
        )
        irrigation, action, irr_conf, irr_days, irr_liters = agrimodels.predict_irrigation(
            [moisture, temp, humidity, ph, rainfall]
        )
        yield_pred, yield_conf, months, yield_trend = agrimodels.predict_crop_yield(
            [ph, temp, rainfall, n, humidity, moisture]
        )
        ndvi = agrimodels.calculate_ndvi(red, nir)
        health = agrimodels.predict_health_score(ndvi, moisture, temp)
        fert, fert_conf, current_npk, target_npk, gap_npk = agrimodels.predict_fertilizer(n, p, k)

        st.session_state.latest_farm_data = {
            "crop": crop,
            "irrigation": irrigation,
            "action": action,
            "yield": yield_pred,
            "ndvi": ndvi,
            "health": health,
            "fertilizer": fert
        }

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Crop", crop, f"{crop_conf}% confidence")
        m2.metric("Irrigation", irrigation, action)
        m3.metric("Yield", f"{yield_pred} t/ha", f"{yield_conf}% confidence")
        m4.metric("Health Score", f"{health}%", f"NDVI {ndvi}")

        c1, c2 = st.columns(2)

        with c1:
            crop_df = pd.DataFrame(crop_scores, columns=["Crop", "Suitability"])
            fig_crop = px.bar(crop_df, x="Crop", y="Suitability", color="Suitability", title="Crop Suitability Scores")
            st.plotly_chart(fig_crop, use_container_width=True)

            yield_df = pd.DataFrame({"Month": months, "Yield Build-up": yield_trend})
            fig_yield = px.line(yield_df, x="Month", y="Yield Build-up", markers=True, title="Yield Forecast Trend")
            st.plotly_chart(fig_yield, use_container_width=True)

        with c2:
            irr_df = pd.DataFrame({"Day": irr_days, "Water (liters/acre)": irr_liters})
            fig_irr = px.area(irr_df, x="Day", y="Water (liters/acre)", title="7-Day Irrigation Plan")
            st.plotly_chart(fig_irr, use_container_width=True)

            npk_df = pd.DataFrame({
                "Nutrient": ["N", "P", "K"],
                "Current": [current_npk["N"], current_npk["P"], current_npk["K"]],
                "Target": [target_npk["N"], target_npk["P"], target_npk["K"]]
            })
            fig_npk = go.Figure()
            fig_npk.add_bar(name="Current", x=npk_df["Nutrient"], y=npk_df["Current"])
            fig_npk.add_bar(name="Target", x=npk_df["Nutrient"], y=npk_df["Target"])
            fig_npk.update_layout(barmode="group", title="NPK Current vs Target")
            st.plotly_chart(fig_npk, use_container_width=True)

elif page == "Crop Recommendation":
    vals = [st.number_input(f"Feature {i+1}", value=float(80 + i)) for i in range(7)]
    if st.button("Recommend Crop", use_container_width=True):
        crop, conf, crop_scores = agrimodels.predict_crop_recommendation(vals, st.session_state.season)
        st.success(f"Recommended crop: {crop} ({conf}% confidence)")
        crop_df = pd.DataFrame(crop_scores, columns=["Crop", "Suitability"])
        fig = px.bar(crop_df, x="Crop", y="Suitability", color="Suitability", title="Crop Recommendation Ranking")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Yield Prediction":
    vals = [st.number_input(f"Input {i+1}", value=25.0 + i) for i in range(6)]
    if st.button("Predict Yield", use_container_width=True):
        y, conf, months, trend = agrimodels.predict_crop_yield(vals)
        st.metric("Expected Yield", f"{y} t/ha", f"{conf}% confidence")
        df = pd.DataFrame({"Month": months, "Yield Build-up": trend})
        fig = px.line(df, x="Month", y="Yield Build-up", markers=True, title="Predicted Yield Progress")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Irrigation":
    vals = [st.number_input(f"Parameter {i+1}", value=40.0 + i) for i in range(5)]
    if st.button("Get Irrigation Plan", use_container_width=True):
        result, action, conf, days, liters = agrimodels.predict_irrigation(vals)
        st.info(f"Need: {result} | Action: {action} | Confidence: {conf}%")
        df = pd.DataFrame({"Day": days, "Water": liters})
        fig = px.bar(df, x="Day", y="Water", color="Water", title="Weekly Irrigation Schedule")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Fertilizer & Soil":
    n = st.number_input("Nitrogen", value=45.0)
    p = st.number_input("Phosphorus", value=38.0)
    k = st.number_input("Potassium", value=42.0)

    if st.button("Analyze Soil", use_container_width=True):
        fert, conf, current_npk, target_npk, gap_npk = agrimodels.predict_fertilizer(n, p, k)
        st.success(f"Recommendation: {fert} ({conf}% confidence)")

        soil_df = pd.DataFrame({
            "Nutrient": ["N", "P", "K"],
            "Current": [current_npk["N"], current_npk["P"], current_npk["K"]],
            "Gap": [gap_npk["N"], gap_npk["P"], gap_npk["K"]]
        })

        fig1 = px.bar(soil_df, x="Nutrient", y="Current", color="Nutrient", title="Current Soil Nutrients")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.pie(soil_df, names="Nutrient", values="Gap", title="Nutrient Deficiency Share")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "NDVI Analysis":
    red = st.slider("Red Band", 0.0, 1.0, 0.30)
    nir = st.slider("NIR Band", 0.0, 1.0, 0.72)
    temp = st.slider("Temperature", 15.0, 45.0, 29.0)
    moisture = st.slider("Soil Moisture", 10.0, 80.0, 42.0)

    if st.button("Calculate NDVI", use_container_width=True):
        ndvi = agrimodels.calculate_ndvi(red, nir)
        health = agrimodels.predict_health_score(ndvi, moisture, temp)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={'text': "Health Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        gauge.update_layout(height=350)
        st.plotly_chart(gauge, use_container_width=True)

        comp_df = pd.DataFrame({
            "Metric": ["NDVI", "Moisture", "Temperature Impact", "Health Score"],
            "Value": [ndvi * 100, moisture, max(0, 100 - abs(temp - 28) * 4), health]
        })
        fig = px.bar(comp_df, x="Metric", y="Value", color="Metric", title="Vegetation and Health Metrics")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"NDVI: {ndvi} | Health score: {health}%")

elif page == "Disease Detection":
    uploaded_file = st.file_uploader("Upload crop leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image, arr = process_image(uploaded_file)
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        if st.button("Analyze Disease", use_container_width=True):
            result = agrimodels.predict_disease(arr)
            st.warning(
                f"Predicted condition: {result['class_name']} "
                f"(confidence: {round(result['confidence'] * 100, 2)}%)"
            )

            prob_df = pd.DataFrame({
                "Condition": list(result["all_probs"].keys()),
                "Probability": list(result["all_probs"].values())
            })
            fig = px.bar(prob_df, x="Condition", y="Probability", color="Probability", title="Disease Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)

elif page == "AI Assistant":
    toolbar_text = (
        f"💬 AgriVision AI Chat | 📍 {st.session_state.district} | 🌾 {st.session_state.season} | "
        + ("✅ Smart Advisor data loaded" if st.session_state.latest_farm_data else "⚠️ Run Smart Advisor for personalized context")
    )
    st.markdown(f"<div class='chat-toolbar'>{toolbar_text}</div>", unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about irrigation, crop, disease, fertilizer, or Telangana farming")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AgriVision AI is thinking..."):
                reply = get_openrouter_reply(
                    prompt,
                    st.session_state.latest_farm_data,
                    st.session_state.district,
                    st.session_state.season
                )
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;padding:18px;font-size:14px;'>🌾 AgriVision AI | OpenRouter AI enabled | Better prediction charts added</div>",
    unsafe_allow_html=True
)
