import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Agri Vision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --bg: #0b1220;
    --panel: rgba(17, 25, 40, 0.82);
    --panel-2: rgba(15, 23, 42, 0.92);
    --border: rgba(148, 163, 184, 0.18);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #14b8a6;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
}

.stApp {
    background: radial-gradient(circle at top left, #123b33 0%, #0b1220 35%, #020617 100%);
    color: var(--text);
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #dffcf6 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1220 100%);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.16), rgba(15, 23, 42, 0.95));
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    color: #ecfeff !important;
}

div[data-testid="stPlotlyChart"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 8px;
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
}

[data-testid="stChatMessage"] {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 8px 10px;
    margin-bottom: 10px;
}

.hero {
    background: linear-gradient(135deg, rgba(20,184,166,0.16), rgba(34,197,94,0.10), rgba(15,23,42,0.95));
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 24px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.22);
}

.caption-pill {
    display:inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(20,184,166,0.14);
    color: #99f6e4;
    border: 1px solid rgba(45,212,191,0.18);
    font-size: 0.85rem;
    margin-bottom: 10px;
}

.subtitle {
    color: #cbd5e1;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: -8px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌾 Agri Vision AI Modules")
selected_module = st.sidebar.radio(
    "Choose Module",
    [
        "Dashboard",
        "NDVI Analysis",
        "Weather vs Yield",
        "Soil NPK",
        "Risk Gauge",
        "Smart Alerts",
        "Recommendations",
        "Chatbot"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙ Control Center")
crop = st.sidebar.selectbox("Select Crop", ["Sesame", "Paddy", "Maize", "Cotton"])
health_score = st.sidebar.slider("Health Score", 0, 100, 10)
moisture = st.sidebar.slider("Soil Moisture (%)", 0, 100, 24)
ndvi = st.sidebar.slider("Current NDVI", 0.0, 1.0, 0.35, 0.01)
risk_level = "HIGH" if health_score < 30 else "MEDIUM" if health_score < 70 else "LOW"
last_refresh = datetime.now().strftime("%d %b %Y, %I:%M %p")

ndvi_data = pd.DataFrame({
    "Days": list(range(1, 11)),
    "NDVI": [0.60, 0.58, 0.55, 0.50, 0.45, 0.42, 0.40, 0.39, 0.38, ndvi]
})

weather_yield = pd.DataFrame({
    "Rainfall": [50, 80, 100, 120, 140],
    "Yield": [10, 15, 20, 23, 25]
})

npk = pd.DataFrame({
    "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
    "Value": [42, 22, 24]
})

fig_ndvi = px.line(ndvi_data, x="Days", y="NDVI", title="NDVI Trend (Crop Health)", markers=True)
fig_ndvi.update_traces(line_color="#14b8a6", marker=dict(size=8))
fig_ndvi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

fig_weather = px.scatter(
    weather_yield,
    x="Rainfall",
    y="Yield",
    size="Yield",
    color="Yield",
    title="Yield vs Rainfall",
    color_continuous_scale="Viridis",
)
fig_weather.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

fig_npk = px.bar(
    npk,
    x="Nutrient",
    y="Value",
    color="Nutrient",
    title="Soil Nutrient Levels",
    color_discrete_sequence=["#22c55e", "#eab308", "#3b82f6"],
)
fig_npk.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "Health Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#ef4444" if health_score < 30 else "#f59e0b" if health_score < 70 else "#22c55e"},
        'steps': [
            {'range': [0, 30], 'color': 'rgba(239,68,68,0.35)'},
            {'range': [30, 70], 'color': 'rgba(245,158,11,0.35)'},
            {'range': [70, 100], 'color': 'rgba(34,197,94,0.35)'}
        ]
    }
))
fig_gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.session_state.chat_history.append({
        "role": "assistant",
        "message": f"Hello! I am your farm assistant. Current crop is {crop}, health score is {health_score}/100, and risk level is {risk_level}."
    })

st.markdown(f"""
<div class='hero'>
    <div class='caption-pill'>LIVE FARM MONITORING</div>
    <h1>🌾 Agri Vision AI</h1>
    <div class='subtitle'>Farm Health AI Dashboard</div>
    <p style='color:#cbd5e1; font-size:1rem;'>Premium AI dashboard for crop health, soil diagnostics, weather relationship, risk analytics, and smart recommendations. Latest dashboard status refresh: <b>{last_refresh}</b>.</p>
</div>
""", unsafe_allow_html=True)

if selected_module == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌱 Crop", crop)
    col2.metric("📊 Health Score", f"{health_score}/100", delta=f"{health_score-90}")
    col3.metric("⚠ Risk Level", risk_level)
    col4.metric("💧 Soil Moisture", f"{moisture}%")

    row1_col1, row1_col2 = st.columns([1.5, 1])
    with row1_col1:
        st.plotly_chart(fig_ndvi, use_container_width=True)
    with row1_col2:
        st.plotly_chart(fig_gauge, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(fig_weather, use_container_width=True)
    with row2_col2:
        st.plotly_chart(fig_npk, use_container_width=True)

elif selected_module == "NDVI Analysis":
    st.subheader("🌿 NDVI Analysis")
    st.plotly_chart(fig_ndvi, use_container_width=True)
    st.info(f"Current NDVI is {ndvi:.2f}. Lower values indicate reduced crop vigor.")

elif selected_module == "Weather vs Yield":
    st.subheader("🌦 Weather vs Yield")
    st.plotly_chart(fig_weather, use_container_width=True)
    st.info("This graph helps analyze how rainfall variations may influence crop yield.")

elif selected_module == "Soil NPK":
    st.subheader("🧪 Soil Nutrient Analysis")
    st.plotly_chart(fig_npk, use_container_width=True)
    st.info("Nitrogen is higher than phosphorus and potassium, so balanced nutrient planning is needed.")

elif selected_module == "Risk Gauge":
    st.subheader("🔥 Health Risk Gauge")
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.info(f"Current farm risk level is {risk_level} based on health score conditions.")

elif selected_module == "Smart Alerts":
    st.subheader("🚨 Smart Alerts")
    if health_score < 30:
        st.error("🚨 High Risk: Immediate action required")
    if moisture < 30:
        st.warning("⚠ Soil moisture low - irrigation needed")
    if ndvi < 0.4:
        st.warning("⚠ Crop stress detected")
    if health_score >= 70 and moisture >= 40 and ndvi >= 0.6:
        st.success("✅ Crop health looks stable and field conditions are favorable")

elif selected_module == "Recommendations":
    st.subheader("📋 Recommendations")
    recommendations = [
        "Increase irrigation frequency if soil moisture remains below 30%.",
        "Inspect leaves for visible stress symptoms because NDVI is trending down.",
        "Improve nutrient planning with follow-up soil testing for phosphorus and potassium.",
        "Track dashboard values daily to compare recovery after intervention."
    ]
    for item in recommendations:
        st.markdown(f"- {item}")

elif selected_module == "Chatbot":
    st.subheader("🤖 AI Agronomy Chatbot")
    user_input = st.chat_input("Ask about disease, irrigation, NPK, or crop stress...")

    if user_input:
        question = user_input.lower()
        if "irrigation" in question or "water" in question:
            answer = f"Soil moisture is {moisture}%. Start irrigation planning immediately." if moisture < 30 else f"Soil moisture is {moisture}%. Irrigation is moderate priority right now."
        elif "ndvi" in question or "stress" in question:
            answer = f"Current NDVI is {ndvi:.2f}. This indicates crop stress and likely reduced vigor." if ndvi < 0.4 else f"Current NDVI is {ndvi:.2f}. Crop greenness is within a safer range."
        elif "npk" in question or "soil" in question:
            answer = "Nitrogen is strongest at 42, while phosphorus and potassium are lower. Balanced fertilization and soil testing are recommended."
        elif "risk" in question:
            answer = f"Risk level is {risk_level}. The score combines crop health, soil moisture, and NDVI trend behavior."
        else:
            answer = "Based on the dashboard, focus first on moisture management, regular field scouting, and correcting nutrient imbalance to improve crop health."

        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "assistant", "message": answer})

    for chat in st.session_state.chat_history:
        role = chat.get("role", "assistant")
        message = chat.get("message", "")
        with st.chat_message(role):
            st.write(message)v
