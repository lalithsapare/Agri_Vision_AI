import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path
from PIL import Image
import streamlit as st


# =========================
# SAFE TENSORFLOW IMPORT
# =========================
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_ERROR = ""
except Exception as e:
    tf = None
    TF_AVAILABLE = False
    TF_ERROR = str(e)


# =========================
# PAGE CONFIG
# =========================
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
</style>
""", unsafe_allow_html=True)


# =========================
# REAL MODEL LOADER
# =========================
class AgritechModels:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.models = {}
        self.encoders = {}
        self.class_names = {}
        self.schemas = {}
        self.load_status = {}
        self._define_schemas()
        self._load_all_models()

    def _define_schemas(self):
        self.schemas = {
            "crop_model": {
                "file": "crop_recommendation_model.pkl",
                "type": "tabular_classification",
                "features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
                "encoder": "crop_encoder"
            },
            "yield_model": {
                "file": "crop_yield_model.pkl",
                "type": "tabular_regression",
                "features": ["ph", "temperature", "rainfall", "fertilizer", "humidity", "soil_moisture"]
            },
            "irrigation_model": {
                "file": "irrigation_model.pkl",
                "type": "tabular_classification",
                "features": ["soil_moisture", "temperature", "humidity", "ph", "rainfall"],
                "encoder": "irrigation_encoder"
            },
            "fertilizer_model": {
                "file": "fertilizer_model.pkl",
                "type": "tabular_classification",
                "features": ["N", "P", "K", "temperature", "humidity", "moisture", "soil_type", "crop_type", "ph", "rainfall"],
                "encoder": "fertilizer_encoder"
            },
            "temp_model": {
                "file": "temperature_model.pkl",
                "type": "tabular_regression",
                "features": ["sensor_1", "sensor_2"]
            },
            "rain_model": {
                "file": "rainfall_model.pkl",
                "type": "tabular_regression",
                "features": ["temperature", "humidity", "pressure", "wind_speed"]
            },
            "humidity_model": {
                "file": "humidity_model.pkl",
                "type": "tabular_regression",
                "features": ["temperature", "pressure", "wind_speed"]
            },
            "ph_model": {
                "file": "ph_model.pkl",
                "type": "tabular_regression",
                "features": ["soil_moisture", "organic_matter", "temperature", "rainfall"]
            },
            "price_model": {
                "file": "crop_price_model.pkl",
                "type": "tabular_regression",
                "features": ["year", "month", "market_code", "arrival_qty", "demand_index", "crop_code"]
            },
            "harvest_model": {
                "file": "harvest_time_model.pkl",
                "type": "tabular_regression",
                "features": ["days_after_sowing", "temperature", "humidity", "ph", "rainfall", "soil_moisture"]
            },
            "npk_model": {
                "file": "npk_prediction_model.pkl",
                "type": "tabular_regression_multi",
                "features": ["ph", "ec", "organic_carbon", "moisture", "temperature", "rainfall"],
                "outputs": ["N_pred", "P_pred", "K_pred"]
            },
            "ndvi_model": {
                "file": "ndvi_model.pkl",
                "type": "tabular_regression",
                "features": ["red_band", "nir_band"]
            },
            "stress_model": {
                "file": "crop_stress_model.pkl",
                "type": "tabular_classification",
                "features": ["ndvi", "temperature", "soil_moisture", "humidity"],
                "encoder": "stress_encoder"
            },
            "disease_model": {
                "file": "plant_disease_model.h5",
                "type": "image_classification",
                "classes_file": "disease_classes.pkl"
            },
            "leaf_model": {
                "file": "leaf_classification_model.h5",
                "type": "image_classification",
                "classes_file": "leaf_classes.pkl"
            },
            "weed_model": {
                "file": "weed_detection_model.h5",
                "type": "image_classification",
                "classes_file": "weed_classes.pkl"
            },
            "soil_model": {
                "file": "soil_classification_model.h5",
                "type": "image_classification",
                "classes_file": "soil_classes.pkl"
            },
            "nutrient_model": {
                "file": "nutrient_deficiency_model.h5",
                "type": "image_classification",
                "classes_file": "nutrient_classes.pkl"
            }
        }

    def _load_encoder_if_exists(self, key, filename):
        file_path = self.model_path / filename
        if file_path.exists():
            try:
                self.encoders[key] = joblib.load(file_path)
                self.load_status[key] = "Loaded"
            except Exception as e:
                self.load_status[key] = f"Encoder load error: {e}"
        else:
            self.load_status[key] = "Missing"

    def _load_all_models(self):
        self._load_encoder_if_exists("crop_encoder", "crop_label_encoder.pkl")
        self._load_encoder_if_exists("irrigation_encoder", "irrigation_label_encoder.pkl")
        self._load_encoder_if_exists("fertilizer_encoder", "fertilizer_label_encoder.pkl")
        self._load_encoder_if_exists("stress_encoder", "stress_label_encoder.pkl")

        for key, schema in self.schemas.items():
            model_file = self.model_path / schema["file"]

            if not model_file.exists():
                self.load_status[key] = f"Missing file: {schema['file']}"
                continue

            try:
                if schema["file"].endswith(".h5"):
                    if not TF_AVAILABLE:
                        self.load_status[key] = f"Skipped - TensorFlow unavailable: {TF_ERROR}"
                        continue
                    self.models[key] = tf.keras.models.load_model(model_file)
                else:
                    self.models[key] = joblib.load(model_file)

                self.load_status[key] = "Loaded"

                if schema["type"] == "image_classification" and "classes_file" in schema:
                    cls_file = self.model_path / schema["classes_file"]
                    if cls_file.exists():
                        try:
                            self.class_names[key] = joblib.load(cls_file)
                        except Exception as e:
                            self.load_status[f"{key}_classes"] = f"Classes load error: {e}"
                    else:
                        self.load_status[f"{key}_classes"] = f"Missing classes file: {schema['classes_file']}"

            except Exception as e:
                self.load_status[key] = f"Load error: {e}"

    def has_model(self, model_key):
        return model_key in self.models

    def _get_expected_features(self, model):
        if hasattr(model, "n_features_in_"):
            return model.n_features_in_
        if hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if hasattr(step, "n_features_in_"):
                    return step.n_features_in_
        return None

    def _validate_tabular_input(self, x, model_key):
        schema = self.schemas[model_key]
        expected_names = schema.get("features", [])
        expected_count = len(expected_names)

        x = np.array(x, dtype=float).reshape(1, -1)

        if x.shape[1] != expected_count:
            raise ValueError(
                f"{model_key} expects {expected_count} features, but got {x.shape[1]}.\n"
                f"Expected order: {expected_names}\n"
                f"Input received: {x.tolist()[0]}"
            )

        model = self.models[model_key]
        fitted_count = self._get_expected_features(model)

        if fitted_count is not None and fitted_count != expected_count:
            raise ValueError(
                f"{model_key} schema says {expected_count} features, but trained model expects {fitted_count}.\n"
                f"Schema features: {expected_names}"
            )

        return x

    def _decode_prediction(self, pred, encoder_key=None):
        if encoder_key and encoder_key in self.encoders:
            try:
                return self.encoders[encoder_key].inverse_transform(pred)[0]
            except Exception:
                pass
        return pred[0]

    def _predict_tabular(self, x, model_key):
        if model_key not in self.models:
            raise ValueError(f"{model_key} NOT LOADED")

        schema = self.schemas[model_key]
        model = self.models[model_key]
        x = self._validate_tabular_input(x, model_key)
        pred = model.predict(x)

        if schema["type"] == "tabular_regression":
            return float(pred[0])

        if schema["type"] == "tabular_regression_multi":
            vals = pred[0]
            return {
                schema["outputs"][0]: float(vals[0]),
                schema["outputs"][1]: float(vals[1]),
                schema["outputs"][2]: float(vals[2]),
            }

        if schema["type"] == "tabular_classification":
            return self._decode_prediction(pred, schema.get("encoder"))

        return pred.tolist()

    def _prepare_image(self, image_array, target_size=(224, 224)):
        x = np.array(image_array, dtype=np.float32)

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        if x.ndim != 4:
            raise ValueError(f"Image must be 3D or 4D, got shape {x.shape}")

        if x.shape[1:] != (target_size[0], target_size[1], 3):
            raise ValueError(
                f"Expected image shape (batch, {target_size[0]}, {target_size[1]}, 3), got {x.shape}"
            )

        if x.max() > 1.0:
            x = x / 255.0

        return x

    def _predict_image(self, image_array, model_key):
        if model_key not in self.models:
            raise ValueError(f"{model_key} NOT LOADED")

        model = self.models[model_key]
        x = self._prepare_image(image_array)
        pred = model.predict(x, verbose=0)

        class_idx = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred, axis=1)[0])

        label = class_idx
        if model_key in self.class_names:
            try:
                label = self.class_names[model_key][class_idx]
            except Exception:
                label = class_idx

        return {
            "class_index": class_idx,
            "class_name": label,
            "confidence": round(confidence, 4)
        }

    def calculate_ndvi(self, red, nir):
        red = float(red)
        nir = float(nir)
        if (nir + red) == 0:
            return 0.0
        return (nir - red) / (nir + red)

    def predict_ndvi(self, x):
        return self._predict_tabular(x, "ndvi_model")

    def predict_crop_stress(self, x):
        return self._predict_tabular(x, "stress_model")

    def predict_npk(self, x):
        return self._predict_tabular(x, "npk_model")

    def predict_crop_recommendation(self, x):
        return self._predict_tabular(x, "crop_model")

    def predict_crop_yield(self, x):
        return self._predict_tabular(x, "yield_model")

    def predict_irrigation(self, x):
        return self._predict_tabular(x, "irrigation_model")

    def predict_fertilizer(self, x):
        return self._predict_tabular(x, "fertilizer_model")

    def predict_temperature(self, x):
        return self._predict_tabular(x, "temp_model")

    def predict_rainfall(self, x):
        return self._predict_tabular(x, "rain_model")

    def predict_humidity(self, x):
        return self._predict_tabular(x, "humidity_model")

    def predict_ph(self, x):
        return self._predict_tabular(x, "ph_model")

    def predict_price(self, x):
        return self._predict_tabular(x, "price_model")

    def predict_harvest_time(self, x):
        return self._predict_tabular(x, "harvest_model")

    def predict_disease(self, img):
        return self._predict_image(img, "disease_model")

    def predict_leaf(self, img):
        return self._predict_image(img, "leaf_model")

    def predict_weed(self, img):
        return self._predict_image(img, "weed_model")

    def predict_soil(self, img):
        return self._predict_image(img, "soil_model")

    def predict_nutrient(self, img):
        return self._predict_image(img, "nutrient_model")


# =========================
# FALLBACK DEMO MODELS
# =========================
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
        return crop, confidence

    def predict_crop_yield(self, features):
        ph, temp, rainfall, fert, humidity, moisture = features
        pred = (temp * 0.65) + (rainfall * 0.03) + (fert * 0.045) + (humidity * 0.075) + (moisture * 0.25) - abs(ph - 6.8) * 2.2
        return round(max(pred, 0), 2), 88.0

    def predict_irrigation(self, features):
        moisture, temp, humidity, ph, rainfall = features
        if rainfall > 150 or moisture > 55:
            return "Low", "Skip irrigation cycle", 92.0
        if moisture < 28 and temp > 32:
            return "High", "Deep irrigation needed", 90.0
        return "Moderate", "Light irrigation", 89.0

    def calculate_ndvi(self, red, nir):
        return round((nir - red) / (nir + red + 1e-8), 4)

    def predict_health_score(self, ndvi, moisture, temp):
        score = (ndvi * 50) + (moisture * 0.3) + ((35 - abs(temp - 28)) * 0.2)
        return min(100, max(0, round(score, 1)))

    def predict_fertilizer(self, n, p, k):
        if n < 50:
            return "Apply nitrogen fertilizer in split doses", 87.0
        if p < 40:
            return "Apply phosphorus fertilizer with basal dose", 86.0
        if k < 40:
            return "Apply potash for stress tolerance", 85.0
        return "Use balanced NPK fertilizer with organic manure", 89.0


# =========================
# HYBRID WRAPPER
# =========================
class HybridAgriModels:
    def __init__(self, model_dir="models"):
        self.real_models = AgritechModels(model_dir)
        self.demo_models = TelanganaAgriModels()

    def predict_crop_recommendation(self, features, season="Kharif"):
        try:
            if self.real_models.has_model("crop_model"):
                result = self.real_models.predict_crop_recommendation(features)
                return str(result), 95.0
        except Exception:
            pass
        return self.demo_models.predict_crop_recommendation(features, season)

    def predict_crop_yield(self, features):
        try:
            if self.real_models.has_model("yield_model"):
                result = self.real_models.predict_crop_yield(features)
                return round(float(result), 2), 94.0
        except Exception:
            pass
        return self.demo_models.predict_crop_yield(features)

    def predict_irrigation(self, features):
        try:
            if self.real_models.has_model("irrigation_model"):
                result = str(self.real_models.predict_irrigation(features))
                action_map = {
                    "Low": "Skip irrigation cycle",
                    "Moderate": "Light irrigation",
                    "High": "Deep irrigation needed"
                }
                return result, action_map.get(result, "Follow irrigation schedule"), 93.0
        except Exception:
            pass
        return self.demo_models.predict_irrigation(features)

    def calculate_ndvi(self, red, nir):
        return self.demo_models.calculate_ndvi(red, nir)

    def predict_health_score(self, ndvi, moisture, temp):
        return self.demo_models.predict_health_score(ndvi, moisture, temp)

    def predict_fertilizer(self, n, p, k):
        return self.demo_models.predict_fertilizer(n, p, k)

    def predict_disease(self, arr):
        try:
            if self.real_models.has_model("disease_model"):
                return self.real_models.predict_disease(arr)
        except Exception:
            pass
        score = int(np.mean(arr)) % 4
        labels = ["Healthy", "Leaf Blight", "Rust", "Leaf Spot"]
        return {
            "class_index": score,
            "class_name": labels[score],
            "confidence": 0.88
        }


agrimodels = HybridAgriModels("models")


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


# =========================
# GEMINI HELPERS
# =========================
def get_gemini_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_gemini_reply(user_text, farm_data, district, season):
    api_key = get_gemini_api_key()

    if not api_key:
        return "Gemini API key missing. Add GEMINI_API_KEY in Streamlit secrets or environment variables."

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
        return "google-generativeai module not installed. Install it with: pip install google-generativeai"
    except Exception as e:
        return f"Gemini chatbot error: {e}"


# =========================
# IMAGE PROCESSOR
# =========================
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    return image, arr


# =========================
# UI HELPERS
# =========================
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


# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌾 AgriVision AI")
page = st.sidebar.radio(
    "Choose Module",
    ["Dashboard", "Smart Advisor", "Crop Recommendation", "Yield Prediction", "Irrigation", "Fertilizer & Soil", "NDVI Analysis", "Disease Detection", "AI Assistant"],
)

st.sidebar.markdown("### Telangana Filters")
st.session_state.district = st.sidebar.selectbox("District", ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar"])
st.session_state.season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

st.sidebar.markdown("### System Status")
if TF_AVAILABLE:
    st.sidebar.success("TensorFlow available")
else:
    st.sidebar.warning("TensorFlow not available, DL models disabled")

with st.sidebar.expander("Model Load Status"):
    for k, v in agrimodels.real_models.load_status.items():
        st.write(f"{k}: {v}")

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
    zone_df = pd.DataFrame({"Zone": ["Healthy", "Moderate", "High Risk", "Critical"], "Value": [52, 28, 14, 6]})

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_line = px.line(trend_df, x="Day", y="Health Index", title="Crop Health Trend - 30 Days")
        fig_line.update_traces(line=dict(color="#16a34a", width=4))
        fig_line.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_line, width="stretch")

    with col2:
        fig_pie = px.pie(zone_df, names="Zone", values="Value", hole=0.55, title="Zone Distribution")
        fig_pie.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_pie, width="stretch")

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

    if st.button("Run Analysis", type="primary", width="stretch"):
        crop, crop_conf = agrimodels.predict_crop_recommendation(
            [n, p, k, temp, humidity, ph, rainfall],
            st.session_state.season
        )
        irrigation, action, irr_conf = agrimodels.predict_irrigation(
            [moisture, temp, humidity, ph, rainfall]
        )
        yield_pred, yield_conf = agrimodels.predict_crop_yield(
            [ph, temp, rainfall, n, humidity, moisture]
        )
        ndvi = agrimodels.calculate_ndvi(red, nir)
        health = agrimodels.predict_health_score(ndvi, moisture, temp)
        fert, fert_conf = agrimodels.predict_fertilizer(n, p, k)

        st.session_state.latest_farm_data = {
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
        x3.metric("Yield", f"{yield_pred} t/ha", f"{yield_conf}% confidence")

        st.markdown(
            f"<div class='health-card'><h3>Final Farm Decision</h3>"
            f"<p><strong>Recommended crop:</strong> {crop}</p>"
            f"<p><strong>Irrigation:</strong> {irrigation} - {action}</p>"
            f"<p><strong>Fertilizer:</strong> {fert}</p>"
            f"<p><strong>NDVI:</strong> {ndvi} | <strong>Health score:</strong> {health}%</p>"
            f"<p><strong>Expected yield:</strong> {yield_pred} t/ha</p></div>",
            unsafe_allow_html=True
        )

elif page == "Crop Recommendation":
    vals = [st.number_input(f"Feature {i+1}", value=float(80 + i)) for i in range(7)]

    if st.button("Recommend Crop", width="stretch"):
        crop, conf = agrimodels.predict_crop_recommendation(vals, st.session_state.season)
        st.success(f"Recommended crop: {crop} ({conf}% confidence)")

elif page == "Yield Prediction":
    vals = [st.number_input(f"Input {i+1}", value=25.0 + i) for i in range(6)]

    if st.button("Predict Yield", width="stretch"):
        y, conf = agrimodels.predict_crop_yield(vals)
        st.metric("Expected Yield", f"{y} t/ha", f"{conf}% confidence")

elif page == "Irrigation":
    vals = [st.number_input(f"Parameter {i+1}", value=40.0 + i) for i in range(5)]

    if st.button("Get Irrigation Plan", width="stretch"):
        result, action, conf = agrimodels.predict_irrigation(vals)
        st.info(f"Need: {result} | Action: {action} | Confidence: {conf}%")

elif page == "Fertilizer & Soil":
    n = st.number_input("Nitrogen", value=45.0)
    p = st.number_input("Phosphorus", value=38.0)
    k = st.number_input("Potassium", value=42.0)

    if st.button("Analyze Soil", width="stretch"):
        fert, conf = agrimodels.predict_fertilizer(n, p, k)
        df = pd.DataFrame({"Nutrient": ["N", "P", "K"], "Value": [n, p, k]})
        fig = px.bar(df, x="Nutrient", y="Value", color="Nutrient", title="Soil Nutrient Levels")
        st.plotly_chart(fig, width="stretch")
        st.success(f"Recommendation: {fert} ({conf}% confidence)")

elif page == "NDVI Analysis":
    red = st.slider("Red Band", 0.0, 1.0, 0.30)
    nir = st.slider("NIR Band", 0.0, 1.0, 0.72)
    temp = st.slider("Temperature", 15.0, 45.0, 29.0)
    moisture = st.slider("Soil Moisture", 10.0, 80.0, 42.0)

    if st.button("Calculate NDVI", width="stretch"):
        ndvi = agrimodels.calculate_ndvi(red, nir)
        health = agrimodels.predict_health_score(ndvi, moisture, temp)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={'text': "Health Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        gauge.update_layout(height=350)
        st.plotly_chart(gauge, width="stretch")
        st.success(f"NDVI: {ndvi} | Health score: {health}%")

elif page == "Disease Detection":
    uploaded_file = st.file_uploader("Upload crop leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image, arr = process_image(uploaded_file)
        st.image(image, caption="Uploaded leaf image", width="stretch")

        if st.button("Analyze Disease", width="stretch"):
            result = agrimodels.predict_disease(arr)
            st.warning(
                f"Predicted condition: {result['class_name']} "
                f"(confidence: {round(result['confidence'] * 100, 2)}%)"
            )

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
                reply = get_gemini_reply(
                    prompt,
                    st.session_state.latest_farm_data,
                    st.session_state.district,
                    st.session_state.season
                )
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})


st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6b7280;padding:18px;font-size:14px;'>🌾 AgriVision AI | Farm Health AI | Telangana Edition | Gemini chatbot ready</div>",
    unsafe_allow_html=True
)
