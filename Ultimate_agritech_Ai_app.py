import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Optional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #0f1f0f 0%, #1a2f1a 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #14532d 0%, #166534 50%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero-dashboard {background: linear-gradient(90deg, #166534, #22c55e, #16a34a); color: white; padding: 1.8rem 1.5rem; border-radius: 24px; box-shadow: 0 12px 32px rgba(22, 101, 52, 0.3); margin-bottom: 1.5rem;}
.metric-card {background: rgba(255,255,255,0.97); border-radius: 20px; padding: 1.4rem; text-align: center; border: 1px solid #dcfce7; box-shadow: 0 8px 24px rgba(0,0,0,0.12);}
.health-card {background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-left: 6px solid #16a34a; border-radius: 20px; padding: 1.4rem; box-shadow: 0 8px 24px rgba(22,101,52,0.15);}
.chat-toolbar {background: rgba(255,255,255,0.96); border-radius: 16px; padding: 1rem 1.2rem; border: 1px solid #dcfce7; margin-bottom: 1rem; color: #14532d;}
.farm-title {font-size: 2.35rem; font-weight: 800; margin: 0;}
.subtitle {margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.96;}
.small-muted {color: #64748b; font-size: 0.9rem;}
.badge-good {background: #dcfce7; color: #166534; padding: 0.35rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700;}
.model-ok {background:#dcfce7;color:#166534;padding:0.25rem 0.5rem;border-radius:999px;font-size:0.75rem;font-weight:700;}
.model-miss {background:#fee2e2;color:#b91c1c;padding:0.25rem 0.5rem;border-radius:999px;font-size:0.75rem;font-weight:700;}
</style>
""", unsafe_allow_html=True)


def get_gemini_api_key():
    try:
        if "AIzaSyDRXgSxLK1hgvciYqCchUdW9b1FAQjBH9o" in st.secrets:
            return st.secrets["AIzaSyDRXgSxLK1hgvciYqCchUdW9b1FAQjBH9o"]
    except Exception:
        pass
    return os.getenv("AIzaSyDRXgSxLK1hgvciYqCchUdW9b1FAQjBH9o", "").strip()

# [Keep all your existing classes: AgritechModels, TelanganaAgriModels, HybridAgriModels - they're perfect]
class AgritechModels:
    # ... (your existing AgritechModels class - no changes needed)
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.models = {}
        self.encoders = {}
        self.class_names = {}
        self.schemas = {}
        self.load_errors = {}
        self._define_schemas()
        self._load_all_models()

    # ... (rest of your existing AgritechModels methods - all perfect)

class TelanganaAgriModels:
    # ... (your existing TelanganaAgriModels class - perfect)

class HybridAgriModels:
    # ... (your existing HybridAgriModels class - perfect)

agrimodels = HybridAgriModels("models")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_farm_data" not in st.session_state:
    st.session_state.latest_farm_data = None

if "district" not in st.session_state:
    st.session_state.district = "Hyderabad"

if "season" not in st.session_state:
    st.session_state.season = "Kharif"

# ✅ FIXED: Correct get_gemini_reply function
def get_gemini_reply(user_text, farm_data, district, season):
    api_key = get_gemini_api_key()

    if not api_key:
        return "❌ **Gemini API key missing.**\n\n**Add your key in:**\n• `.streamlit/secrets.toml` file\n• Or environment variable `GEMINI_API_KEY`"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        farm_context = "No latest farm data available."

        if farm_data:
            farm_context = (
                f"District: {district}. Season: {season}. Recommended crop: {farm_data.get('crop')}. "
                f"Irrigation: {farm_data.get('irrigation')} ({farm_data.get('action')}). "
                f"Expected yield: {farm_data.get('yield')} t/ha. NDVI: {farm_data.get('ndvi')}. "
                f"Health score: {farm_data.get('health')}%. Fertilizer: {farm_data.get('fertilizer')}."
            )

        prompt = f"""
You are AgriVision AI, a practical Telangana agriculture assistant.

Answer in simple farmer-friendly language.

Structure reply as:
1. What is happening
2. Why  
3. Immediate action
4. Next 3 days

Farm context:
{farm_context}

User question:
{user_text}
"""
        response = model.generate_content(prompt)
        return response.text.strip()

    except ModuleNotFoundError:
        return "❌ Install: `pip install google-generativeai`"
    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

# [Keep all your existing helper functions - perfect]
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

# [Keep ALL your existing UI code - it's perfect as-is]
# Sidebar, page selection, all pages ("Dashboard", "Smart Advisor", etc.) - NO CHANGES NEEDED

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

st.sidebar.markdown("### Telangana Filters")
st.session_state.district = st.sidebar.selectbox(
    "District",
    ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar"]
)
st.session_state.season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

st.sidebar.markdown("### Model Engine")
if TF_AVAILABLE:
    st.sidebar.markdown("<span class='model-ok'>TensorFlow optional support ON</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span class='model-miss'>TensorFlow unavailable, .h5 skipped</span>", unsafe_allow_html=True)

with st.sidebar.expander("Loaded Models Status"):
    status_df = agrimodels.model_status()
    st.dataframe(status_df, use_container_width=True, hide_index=True)

render_header()
render_top_dashboard()

# [Keep ALL your existing page logic - perfect]
if page == "Dashboard":
    # ... your existing dashboard code
    pass
elif page == "Smart Advisor":
    # ... your existing smart advisor code  
    pass
# ... (all other pages exactly as you wrote them)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6b7280;padding:18px;font-size:14px;'>🌾 AgriVision AI | Farm Health AI | Telangana Edition | Gemini chatbot ready</div>",
    unsafe_allow_html=True
)
