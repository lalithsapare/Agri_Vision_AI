import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSense AI | Smart Farm Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..700&family=Sora:wght@600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2e1f 0%, #0d3b28 100%);
        color: #d6f0e0;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #d6f0e0 !important; }

    .main-header {
        background: linear-gradient(135deg, #0a2e1f 0%, #1b5e38 60%, #2d7a50 100%);
        padding: 2rem 2.5rem 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(10,46,31,0.25);
    }
    .main-header h1 { font-family: 'Sora', sans-serif; font-size: 2.2rem; margin: 0; }
    .main-header p  { opacity: 0.82; margin: 0.4rem 0 0; font-size: 1rem; }

    .module-card {
        background: #f7f9f7;
        border: 1px solid #cde3d5;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .module-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.2rem;
        color: #0a2e1f;
        margin-bottom: 0.4rem;
    }

    .result-healthy {
        background: #e8f5e9; border-left: 4px solid #43a047;
        padding: 1rem 1.2rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-disease {
        background: #fff3e0; border-left: 4px solid #fb8c00;
        padding: 1rem 1.2rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-severe {
        background: #fce4ec; border-left: 4px solid #e53935;
        padding: 1rem 1.2rem; border-radius: 8px; margin-top: 1rem;
    }

    .confidence-badge {
        display: inline-block; background: #0a2e1f; color: white;
        padding: 0.25rem 0.85rem; border-radius: 999px;
        font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem;
    }

    .stFileUploader { border: 2px dashed #1b5e38 !important; border-radius: 10px; }

    [data-testid="metric-container"] {
        background: #f0f7f3; border-radius: 10px; padding: 0.8rem;
        border: 1px solid #c5ddd0;
    }

    .status-pill {
        display:inline-block; padding:0.2rem 0.7rem;
        border-radius:999px; font-size:0.78rem; font-weight:600;
        margin-left:0.5rem;
    }
    .pill-ok  { background:#e8f5e9; color:#2e7d32; }
    .pill-err { background:#fce4ec; color:#c62828; }

    div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_joblib(path: str):
    try:
        obj = joblib.load(path)
        return obj, None
    except Exception as e:
        return None, str(e)

def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img

def predict_top_class(model, class_dict: dict, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]
    if isinstance(next(iter(class_dict)), str):
        idx_to_label = {v: k for k, v in class_dict.items()}
    else:
        idx_to_label = class_dict

    top_idx = int(np.argmax(preds))
    top_label = idx_to_label.get(top_idx, f"Class {top_idx}")
    confidence = float(preds[top_idx]) * 100
    all_probs = {
        idx_to_label.get(i, f"Class {i}"): float(p) * 100
        for i, p in enumerate(preds)
    }
    return top_label, confidence, all_probs

def result_box(label: str, confidence: float, label_type: str = "auto"):
    lwr = label.lower()
    if label_type == "auto":
        if "healthy" in lwr:
            box_cls = "result-healthy"
        elif confidence < 55:
            box_cls = "result-disease"
        else:
            box_cls = "result-severe"
    else:
        box_cls = {
            "healthy": "result-healthy",
            "warning": "result-disease",
            "danger": "result-severe"
        }.get(label_type, "result-disease")

    st.markdown(f"""
    <div class="{box_cls}">
        <strong>🔍 Detected:</strong> {label}<br>
        <span class="confidence-badge">Confidence: {confidence:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

def model_status_pill(model_obj, path: str):
    if model_obj is not None:
        return '<span class="status-pill pill-ok">✓ Loaded</span>'
    return f'<span class="status-pill pill-err">✗ {os.path.basename(path)} not found</span>'

# ─────────────────────────────────────────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    "tomato_disease_model": "tomato_disease_model.h5",
    "tomato_classes": "models/tomato_classes.joblib",
    "rice_disease_model": "rice_disease_model.h5",
    "rice_classes": "rice_classes.joblib",
    "rice_pest_model": "rice_pest_model.h5",
    "rice_pest_classes": "rice_pest_classes.joblib",
    "nutrient_model": "nutrient_deficiency_cnn_model.keras",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading AI models..."):
    tomato_disease_model, _tde = load_keras_model(PATHS["tomato_disease_model"])
    tomato_classes, _tce = load_joblib(PATHS["tomato_classes"])
    rice_disease_model, _rde = load_keras_model(PATHS["rice_disease_model"])
    rice_classes, _rce = load_joblib(PATHS["rice_classes"])
    rice_pest_model, _rpe = load_keras_model(PATHS["rice_pest_model"])
    rice_pest_classes, _rpce = load_joblib(PATHS["rice_pest_classes"])
    nutrient_model, _nde = load_keras_model(PATHS["nutrient_model"])

DEFAULT_NUTRIENT_CLASSES = {
    0: "Nitrogen Deficiency",
    1: "Phosphorus Deficiency",
    2: "Potassium Deficiency",
    3: "Iron Deficiency",
    4: "Magnesium Deficiency",
    5: "Calcium Deficiency",
    6: "Sulfur Deficiency",
    7: "Healthy / No Deficiency",
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 AgroSense AI")
    st.markdown("### Navigation")
    module = st.selectbox(
        "Select Detection Module",
        [
            "🏠 Dashboard",
            "🍅 Tomato Disease Detection",
            "🌾 Rice Disease Detection",
            "🐛 Rice Pest Detection",
            "🧪 Nutrient Deficiency Detection",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    st.markdown(
        f"Tomato Disease {model_status_pill(tomato_disease_model, PATHS['tomato_disease_model'])}<br>"
        f"Tomato Classes {model_status_pill(tomato_classes, PATHS['tomato_classes'])}<br>"
        f"Rice Disease {model_status_pill(rice_disease_model, PATHS['rice_disease_model'])}<br>"
        f"Rice Classes {model_status_pill(rice_classes, PATHS['rice_classes'])}<br>"
        f"Rice Pest {model_status_pill(rice_pest_model, PATHS['rice_pest_model'])}<br>"
        f"Pest Classes {model_status_pill(rice_pest_classes, PATHS['rice_pest_classes'])}<br>"
        f"Nutrient CNN {model_status_pill(nutrient_model, PATHS['nutrient_model'])}",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📍 Location")
    st.markdown("Hyderabad, Telangana")
    st.markdown("### ℹ️ About")
    st.markdown(
        "AgroSense AI integrates multiple deep-learning models for crop disease, pest, "
        "and nutrient deficiency detection tailored to Telangana agri-conditions."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
MODULE_META = {
    "🏠 Dashboard": (
        "🌿 AgroSense AI — Smart Farm Intelligence",
        "AI-powered crop disease, pest & nutrient deficiency detection platform",
    ),
    "🍅 Tomato Disease Detection": (
        "🍅 Tomato Disease Detection",
        "Upload a tomato leaf image to detect diseases using CNN",
    ),
    "🌾 Rice Disease Detection": (
        "🌾 Rice Disease Detection",
        "Upload a rice leaf image to identify common rice diseases",
    ),
    "🐛 Rice Pest Detection": (
        "🐛 Rice Pest Detection",
        "Upload a rice crop image to detect pest infestations",
    ),
    "🧪 Nutrient Deficiency Detection": (
        "🧪 Nutrient Deficiency Detection",
        "Upload a crop leaf image to diagnose nutrient deficiencies",
    ),
}

title, subtitle = MODULE_META[module]
st.markdown(f"""
<div class="main-header">
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Generic Detection UI
# ─────────────────────────────────────────────────────────────────────────────
def detection_ui(model, class_dict, module_key: str, info_map: dict = None, target_size=(224, 224)):
    if model is None or class_dict is None:
        st.error(
            "⚠️ Required model/class file not found. Please ensure the model and joblib files "
            "exist in the correct paths."
        )
        with st.expander("📁 Expected file paths"):
            for _, v in PATHS.items():
                st.code(v)
        return

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📤 Upload Leaf / Crop Image")
        uploaded = st.file_uploader(
            "Choose an image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"uploader_{module_key}",
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with col_result:
        st.markdown("#### 🔬 Analysis Result")
        if not uploaded:
            st.info("⬅️ Upload an image to run detection.")
            return

        with st.spinner("🤖 Running inference..."):
            try:
                img_array, pil_img = preprocess_image(uploaded, target_size)
                label, confidence, all_probs = predict_top_class(model, class_dict, img_array)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        result_box(label, confidence)

        if info_map and label in info_map:
            info = info_map[label]
            st.markdown("---")
            st.markdown(f"**📋 About:** {label}")
            if "cause" in info:
                st.markdown(f"- **Cause:** {info['cause']}")
            if "severity" in info:
                st.markdown(f"- **Severity:** {info['severity']}")
            if "treatment" in info:
                st.markdown(f"- **Treatment:** {info['treatment']}")

        st.markdown("---")
        st.markdown("**📊 Top Predictions**")
        top5 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for lbl, prob in top5:
            st.progress(int(prob), text=f"{lbl}: {prob:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Info Maps
# ─────────────────────────────────────────────────────────────────────────────
TOMATO_INFO = {
    "Tomato_Bacterial_spot": {
        "cause": "Xanthomonas campestris pv. vesicatoria",
        "severity": "⚠️ Moderate – High",
        "treatment": "Copper-based bactericides; remove infected leaves; crop rotation.",
    },
    "Tomato_Early_blight": {
        "cause": "Alternaria solani fungus",
        "severity": "⚠️ Moderate",
        "treatment": "Apply mancozeb or chlorothalonil fungicide; avoid overhead watering.",
    },
    "Tomato_Late_blight": {
        "cause": "Phytophthora infestans oomycete",
        "severity": "🔴 High – Can destroy entire crop",
        "treatment": "Metalaxyl + mancozeb spray; destroy infected plants immediately.",
    },
    "Tomato_Leaf_Mold": {
        "cause": "Passalora fulva fungus",
        "severity": "⚠️ Moderate",
        "treatment": "Improve air circulation; apply chlorothalonil fungicide.",
    },
    "Tomato_Mosaic_virus": {
        "cause": "Tomato mosaic virus (ToMV)",
        "severity": "🔴 High – No chemical cure",
        "treatment": "Remove infected plants; use resistant varieties; control aphids.",
    },
    "Tomato_healthy": {
        "cause": "N/A",
        "severity": "✅ Healthy",
        "treatment": "No treatment required. Maintain good agronomic practices.",
    },
}

RICE_DISEASE_INFO = {
    "Bacterial_leaf_blight": {
        "cause": "Xanthomonas oryzae pv. oryzae",
        "severity": "🔴 High",
        "treatment": "Copper oxychloride spray; resistant varieties like Samba Mahsuri.",
    },
    "Brown_spot": {
        "cause": "Helminthosporium oryzae",
        "severity": "⚠️ Moderate",
        "treatment": "Mancozeb 2.5 g/L spray; balanced N-P-K fertilisation.",
    },
    "Blast": {
        "cause": "Magnaporthe oryzae",
        "severity": "🔴 High",
        "treatment": "Tricyclazole 0.06% spray; avoid excess nitrogen.",
    },
    "Healthy": {
        "cause": "N/A",
        "severity": "✅ Healthy",
        "treatment": "No treatment required.",
    },
}

RICE_PEST_INFO = {
    "Stem_borer": {
        "cause": "Scirpophaga incertulas larvae",
        "severity": "🔴 High – Dead heart / White ear",
        "treatment": "Cartap hydrochloride 4G @ 20 kg/ha; light traps for moths.",
    },
    "Brown_plant_hopper": {
        "cause": "Nilaparvata lugens",
        "severity": "🔴 High – Hopperburn",
        "treatment": "Imidacloprid 200 SL @ 100 mL/ha; avoid excess N.",
    },
    "Leaf_folder": {
        "cause": "Cnaphalocrocis medinalis larvae",
        "severity": "⚠️ Moderate",
        "treatment": "Chlorpyrifos 20 EC @ 1.5 L/ha spray; drain field periodically.",
    },
    "Healthy": {
        "cause": "N/A",
        "severity": "✅ No pest detected",
        "treatment": "No action required.",
    },
}

NUTRIENT_INFO = {
    "Nitrogen Deficiency": {
        "cause": "Low soil N or waterlogged, poorly aerated soil",
        "severity": "⚠️ Moderate – affects yield significantly",
        "treatment": "Apply urea (46% N) @ 30–50 kg/ha as top dressing.",
    },
    "Phosphorus Deficiency": {
        "cause": "Acidic soils, poor root development",
        "severity": "⚠️ Moderate",
        "treatment": "Single superphosphate 100 kg/ha basal application.",
    },
    "Potassium Deficiency": {
        "cause": "Sandy or light soils with leaching",
        "severity": "⚠️ Moderate",
        "treatment": "Muriate of potash 50 kg/ha; foliar KNO₃ spray.",
    },
    "Iron Deficiency": {
        "cause": "Alkaline soils with high pH (>7.5)",
        "severity": "⚠️ Moderate – causes interveinal chlorosis",
        "treatment": "FeSO₄ 0.5% foliar spray; soil acidification.",
    },
    "Magnesium Deficiency": {
        "cause": "Acidic or sandy soils",
        "severity": "⚠️ Moderate",
        "treatment": "Dolomite lime or MgSO₄ 10 kg/ha spray.",
    },
    "Calcium Deficiency": {
        "cause": "Acidic soils or low Ca availability",
        "severity": "⚠️ Moderate",
        "treatment": "Lime application; foliar CaCl₂ 0.5% spray.",
    },
    "Sulfur Deficiency": {
        "cause": "Low organic matter soils",
        "severity": "⚠️ Moderate",
        "treatment": "Gypsum (CaSO₄) 200 kg/ha; single superphosphate.",
    },
    "Healthy / No Deficiency": {
        "cause": "N/A",
        "severity": "✅ Healthy – Nutrients balanced",
        "treatment": "Continue current fertilisation schedule.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if module == "🏠 Dashboard":
    st.markdown("### 📊 System Overview")
    c1, c2, c3, c4 = st.columns(4)

    models_loaded = sum([
        tomato_disease_model is not None,
        rice_disease_model is not None,
        rice_pest_model is not None,
        nutrient_model is not None,
    ])

    c1.metric("🤖 AI Models Loaded", f"{models_loaded}/4")
    c2.metric("🍅 Tomato Classes", len(tomato_classes) if tomato_classes else "N/A")
    c3.metric("🌾 Rice Disease Classes", len(rice_classes) if rice_classes else "N/A")
    c4.metric("🐛 Rice Pest Classes", len(rice_pest_classes) if rice_pest_classes else "N/A")

    st.markdown("---")
    st.markdown("### 🧩 Available Detection Modules")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🍅 Tomato Disease Detection</div>
            <p style="color:#4a6355;font-size:0.9rem;">
                Detects bacterial spot, early blight, late blight, leaf mold,
                mosaic virus, septoria leaf spot, spider mites, target spot,
                yellow leaf curl, and healthy tomato leaves using
                <strong>tomato_disease_model.h5</strong>.
            </p>
        </div>
        <div class="module-card">
            <div class="module-title">🐛 Rice Pest Detection</div>
            <p style="color:#4a6355;font-size:0.9rem;">
                Identifies rice pests including stem borer, brown plant hopper,
                leaf folder, and others using <strong>rice_pest_model.h5</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🌾 Rice Disease Detection</div>
            <p style="color:#4a6355;font-size:0.9rem;">
                Identifies bacterial leaf blight, blast, brown spot, sheath blight,
                and tungro using <strong>rice_disease_model.h5</strong>.
            </p>
        </div>
        <div class="module-card">
            <div class="module-title">🧪 Nutrient Deficiency Detection</div>
            <p style="color:#4a6355;font-size:0.9rem;">
                Diagnoses nitrogen, phosphorus, potassium, iron, magnesium,
                calcium, and sulfur deficiencies using
                <strong>nutrient_deficiency_cnn_model.keras</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Select a detection module from the sidebar to get started.")

# ─────────────────────────────────────────────────────────────────────────────
# Tomato Disease Detection
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🍅 Tomato Disease Detection":
    st.markdown("""
    > **Model:** `tomato_disease_model.h5` | **Classes:** `models/tomato_classes.joblib`
    """)
    detection_ui(
        model=tomato_disease_model,
        class_dict=tomato_classes,
        module_key="tomato_disease",
        info_map=TOMATO_INFO,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Rice Disease Detection
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🌾 Rice Disease Detection":
    st.markdown("""
    > **Model:** `rice_disease_model.h5` | **Classes:** `rice_classes.joblib`
    """)
    detection_ui(
        model=rice_disease_model,
        class_dict=rice_classes,
        module_key="rice_disease",
        info_map=RICE_DISEASE_INFO,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Rice Pest Detection
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🐛 Rice Pest Detection":
    st.markdown("""
    > **Model:** `rice_pest_model.h5` | **Classes:** `rice_pest_classes.joblib`
    """)
    detection_ui(
        model=rice_pest_model,
        class_dict=rice_pest_classes,
        module_key="rice_pest",
        info_map=RICE_PEST_INFO,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Nutrient Deficiency Detection
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🧪 Nutrient Deficiency Detection":
    st.markdown("""
    > **Model:** `nutrient_deficiency_cnn_model.keras` | **Classes:** default nutrient class map
    """)
    nutrient_classes = DEFAULT_NUTRIENT_CLASSES
    detection_ui(
        model=nutrient_model,
        class_dict=nutrient_classes,
        module_key="nutrient",
        info_map=NUTRIENT_INFO,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#7a9e8a;font-size:0.82rem;'>"
    "AgroSense AI • Built for Telangana Farmers • Powered by TensorFlow/Keras + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
