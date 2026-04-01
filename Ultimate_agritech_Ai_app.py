import random
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Agri Vision AI – Telangana Edition", page_icon="🌾", layout="wide")

APP_NAME = "Agri Vision AI – Telangana Edition"

TELANGANA_CROPS = {
    "Kharif": ["Cotton", "Maize", "Paddy", "Red Gram", "Soybean", "Green Gram", "Black Gram", "Jowar", "Bajra", "Sesame", "Groundnut", "Turmeric", "Chilli"],
    "Rabi": ["Paddy", "Maize", "Groundnut", "Bengal Gram", "Black Gram", "Sesame", "Jowar", "Sunflower", "Safflower", "Horse Gram", "Castor"],
    "Summer": ["Paddy", "Maize", "Groundnut", "Sesame", "Sunflower", "Vegetables", "Fodder Maize"],
}

TELANGANA_DISTRICTS = [
    "Adilabad", "Bhadradri Kothagudem", "Hanamkonda", "Hyderabad", "Jagtial", "Jangaon",
    "Jayashankar Bhupalpally", "Jogulamba Gadwal", "Kamareddy", "Karimnagar", "Khammam",
    "Komaram Bheem Asifabad", "Mahabubabad", "Mahabubnagar", "Mancherial", "Medak",
    "Medchal-Malkajgiri", "Mulugu", "Nagarkurnool", "Nalgonda", "Narayanpet", "Nirmal",
    "Nizamabad", "Peddapalli", "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet",
    "Suryapet", "Vikarabad", "Wanaparthy", "Warangal", "Yadadri Bhuvanagiri"
]

SOIL_TYPES = ["Red Sandy Loam", "Black Cotton Soil", "Alluvial Soil", "Lateritic Soil", "Clay Loam"]
FARM_TYPES = ["Dry land", "Irrigated", "Rain-fed"]

SIMULATION_DEFAULTS = {
    "Dry land": {"soil_moisture": 24.0, "rainfall": 60.0, "humidity": 42.0, "temperature": 33.0, "nitrogen": 42, "phosphorus": 22, "potassium": 24, "ph": 7.2, "ndvi": 0.39},
    "Irrigated": {"soil_moisture": 71.0, "rainfall": 220.0, "humidity": 70.0, "temperature": 28.0, "nitrogen": 74, "phosphorus": 39, "potassium": 44, "ph": 6.8, "ndvi": 0.78},
    "Rain-fed": {"soil_moisture": 49.0, "rainfall": 510.0, "humidity": 76.0, "temperature": 29.0, "nitrogen": 58, "phosphorus": 31, "potassium": 34, "ph": 6.9, "ndvi": 0.58},
}

CROP_GUIDE = {
    "Cotton": {"water": "Medium", "soil": "Black Cotton Soil", "base_yield": 18, "fert": "Balanced NPK with split nitrogen application"},
    "Maize": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 32, "fert": "Nitrogen rich basal + top dressing"},
    "Paddy": {"water": "High", "soil": "Clay Loam", "base_yield": 26, "fert": "High nitrogen with zinc attention"},
    "Red Gram": {"water": "Low-Medium", "soil": "Red Sandy Loam", "base_yield": 10, "fert": "Starter phosphorus, low nitrogen"},
    "Soybean": {"water": "Medium", "soil": "Alluvial Soil", "base_yield": 14, "fert": "Rhizobium support + phosphorus"},
    "Green Gram": {"water": "Low", "soil": "Red Sandy Loam", "base_yield": 8, "fert": "Light phosphorus and sulphur"},
    "Black Gram": {"water": "Low", "soil": "Red Sandy Loam", "base_yield": 8, "fert": "Low nitrogen, phosphorus support"},
    "Jowar": {"water": "Low", "soil": "Black Cotton Soil", "base_yield": 14, "fert": "Moderate NPK with micronutrient watch"},
    "Bajra": {"water": "Low", "soil": "Lateritic Soil", "base_yield": 12, "fert": "Low input crop, moderate nitrogen"},
    "Sesame": {"water": "Low", "soil": "Red Sandy Loam", "base_yield": 6, "fert": "Sulphur and phosphorus support"},
    "Groundnut": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 16, "fert": "Gypsum, calcium and phosphorus"},
    "Turmeric": {"water": "High", "soil": "Clay Loam", "base_yield": 80, "fert": "Organic matter rich with potassium support"},
    "Chilli": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 22, "fert": "Potash and micronutrient support"},
    "Bengal Gram": {"water": "Low", "soil": "Black Cotton Soil", "base_yield": 10, "fert": "Phosphorus dominant recommendation"},
    "Sunflower": {"water": "Low-Medium", "soil": "Alluvial Soil", "base_yield": 9, "fert": "Sulphur and boron support"},
    "Safflower": {"water": "Low", "soil": "Black Cotton Soil", "base_yield": 7, "fert": "Low irrigation, balanced phosphorus"},
    "Horse Gram": {"water": "Low", "soil": "Lateritic Soil", "base_yield": 6, "fert": "Very low input, organic matter preferred"},
    "Castor": {"water": "Low-Medium", "soil": "Black Cotton Soil", "base_yield": 11, "fert": "Moderate NPK with sulphur"},
    "Vegetables": {"water": "Medium-High", "soil": "Clay Loam", "base_yield": 95, "fert": "Crop-specific soluble nutrition"},
    "Fodder Maize": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 140, "fert": "Nitrogen rich fodder nutrition"},
}

DISEASE_RULES = {
    "Paddy": [(0.45, 78, "Leaf Blight", 88), (0.60, 70, "Blast Risk", 81)],
    "Cotton": [(0.40, 65, "Leaf Spot", 84), (0.55, 75, "Wilt Risk", 76)],
    "Maize": [(0.42, 70, "Leaf Blight", 85)],
    "Chilli": [(0.48, 72, "Leaf Curl Risk", 82)],
    "Groundnut": [(0.43, 68, "Tikka Disease Risk", 80)],
}


def recommend_crop(season, soil_type, temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium):
    scored = []
    for crop in TELANGANA_CROPS[season]:
        score = 50
        guide = CROP_GUIDE.get(crop, {})
        if guide.get("soil") == soil_type:
            score += 18
        if crop == "Cotton" and 24 <= temperature <= 32 and rainfall >= 500:
            score += 24
        if crop == "Paddy" and humidity >= 70 and rainfall >= 600:
            score += 25
        if crop == "Maize" and 20 <= temperature <= 32 and 400 <= rainfall <= 900:
            score += 22
        if crop == "Red Gram" and 22 <= temperature <= 34 and rainfall >= 450:
            score += 20
        if crop == "Groundnut" and 22 <= temperature <= 30 and 300 <= rainfall <= 700:
            score += 20
        if crop == "Bengal Gram" and season == "Rabi" and rainfall <= 150:
            score += 25
        if crop == "Sesame" and rainfall <= 500:
            score += 17
        if crop == "Turmeric" and humidity >= 75 and rainfall >= 800:
            score += 18
        if 6.0 <= ph <= 7.8:
            score += 8
        if nitrogen >= 50:
            score += 5
        if phosphorus >= 25:
            score += 5
        if potassium >= 25:
            score += 5
        confidence = min(97, max(68, score))
        scored.append({"crop": crop, "score": score, "confidence": confidence})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]


def predict_yield(crop, area, rainfall, temperature, soil_moisture, nitrogen, phosphorus, potassium):
    base = CROP_GUIDE.get(crop, {}).get("base_yield", 12)
    weather_factor = 1 + ((rainfall - 300) / 2000) + ((soil_moisture - 40) / 300)
    nutrient_factor = 1 + ((nitrogen + phosphorus + potassium) - 100) / 500
    temp_penalty = 1 - abs(28 - temperature) / 60
    per_acre = max(3, base * weather_factor * nutrient_factor * temp_penalty)
    total = per_acre * area
    confidence = max(72, min(96, int(78 + (soil_moisture / 10) + (nitrogen + phosphorus + potassium) / 20)))
    return round(per_acre, 2), round(total, 2), confidence


def predict_disease(crop, ndvi, humidity, soil_moisture):
    rules = DISEASE_RULES.get(crop, [])
    for ndvi_limit, humidity_limit, disease_name, conf in rules:
        if ndvi <= ndvi_limit and humidity >= humidity_limit:
            return disease_name, conf
    if soil_moisture < 25 and ndvi < 0.45:
        return "Stress Wilt Risk", 79
    return "Healthy", 91


def weather_label(rainfall, humidity):
    if rainfall >= 700:
        return "High Rainfall"
    if rainfall >= 300:
        return "Moderate Rainfall"
    if humidity < 40:
        return "Dry Weather"
    return "Low Rainfall"


def irrigation_advice(crop, soil_moisture, temperature, rainfall, disease_name):
    if disease_name != "Healthy":
        return "Low", "Reduce irrigation and avoid excess leaf wetness because disease risk is active."
    if soil_moisture < 30 or (temperature > 34 and rainfall < 50):
        return "High", "Immediate irrigation needed; prefer drip or early morning watering."
    if soil_moisture < 45:
        return "Medium", "Irrigation needed in 1-2 days; protect crop at flowering and grain filling stage."
    return "Low", "Moisture is acceptable; avoid over-irrigation in Telangana field conditions."


def fertilizer_advice(crop, nitrogen, phosphorus, potassium, disease_name):
    low = []
    if nitrogen < 50:
        low.append("nitrogen")
    if phosphorus < 25:
        low.append("phosphorus")
    if potassium < 25:
        low.append("potassium")
    if disease_name != "Healthy":
        return "Reduced usage", "Disease detected, so avoid heavy fertilizer immediately. Use balanced corrective nutrition only after crop condition improves."
    if not low:
        return "Balanced usage", f"NPK looks acceptable for {crop}. Follow split doses and crop-stage based application."
    return "Targeted usage", f"Low {', '.join(low)} detected. For {crop}: {CROP_GUIDE.get(crop, {}).get('fert', 'Balanced fertilization recommended')}."


def nutrient_health(nitrogen, phosphorus, potassium):
    avg = (nitrogen + phosphorus + potassium) / 3
    if avg >= 55:
        return "Strong"
    if avg >= 35:
        return "Moderate"
    return "Weak"


def ndvi_status(ndvi):
    if ndvi >= 0.70:
        return "Healthy canopy"
    if ndvi >= 0.50:
        return "Moderate canopy"
    return "Weak canopy"


def build_intelligence_layer(crop, disease_name, ndvi, soil_moisture, rainfall, temperature, nitrogen, phosphorus, potassium):
    irrigation_level, irrigation_text = irrigation_advice(crop, soil_moisture, temperature, rainfall, disease_name)
    fertilizer_level, fertilizer_text = fertilizer_advice(crop, nitrogen, phosphorus, potassium, disease_name)

    if disease_name != "Healthy":
        fertilizer_level = "Reduced usage"
        irrigation_level = "Low"
        irrigation_text = "Reduce irrigation because disease pressure is active and excess moisture can worsen infection."
        fertilizer_text = "Reduce fertilizer usage temporarily and avoid heavy nitrogen until disease pressure comes down."

    return {
        "irrigation_level": irrigation_level,
        "irrigation_text": irrigation_text,
        "fertilizer_level": fertilizer_level,
        "fertilizer_text": fertilizer_text,
    }


def calculate_risk_and_health(disease_name, ndvi, soil_moisture, nutrient_state):
    risk_score = 0
    if disease_name != "Healthy":
        risk_score += 50
    if ndvi < 0.5:
        risk_score += 30
    if soil_moisture < 30:
        risk_score += 10
    if nutrient_state == "Weak":
        risk_score += 10

    if risk_score > 60:
        risk = "HIGH"
    elif risk_score >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    health_score = max(10, 100 - risk_score)
    return risk_score, risk, health_score


def build_smart_farm_report(location, season, crop, crop_conf, yield_total, yield_conf, weather_name, disease_name, disease_conf, risk, health_score, intelligence):
    actions = []
    if intelligence["irrigation_level"] == "Low":
        actions.append("Reduce irrigation")
    elif intelligence["irrigation_level"] == "High":
        actions.append("Increase irrigation immediately")
    else:
        actions.append("Monitor irrigation in 1-2 days")

    if disease_name != "Healthy":
        actions.append("Apply fungicide or crop-protection spray after field inspection")

    if intelligence["fertilizer_level"] == "Reduced usage":
        actions.append("Avoid heavy nitrogen fertilizer now")
    elif intelligence["fertilizer_level"] == "Targeted usage":
        actions.append("Add balanced nitrogen-phosphorus-potassium correction")
    else:
        actions.append("Continue split fertilizer schedule")

    return {
        "location": location,
        "season": season,
        "crop": crop,
        "crop_conf": crop_conf,
        "yield_total": yield_total,
        "yield_conf": yield_conf,
        "weather": weather_name,
        "disease": disease_name,
        "disease_conf": disease_conf,
        "risk": risk,
        "health_score": health_score,
        "actions": actions[:3],
    }


def build_chatbot_reply(question, report, intelligence, ndvi_value, nutrient_state):
    q = question.lower().strip()
    intro = f"🤖 Agri Vision AI Assistant: Based on your Telangana farm report, crop is {report['crop']}, risk level is {report['risk']}, and health score is {report['health_score']}/100."
    if "what should i do" in q or "what to do" in q:
        return f"{intro}\n\nRecommended actions: {', '.join(report['actions'])}. Main reason: disease status is {report['disease']} and NDVI is {ndvi_value:.2f}, so crop stress needs immediate management."
    if "irrigation" in q or "water" in q:
        return f"{intro}\n\nIrrigation advice: {intelligence['irrigation_text']}"
    if "fertilizer" in q or "npk" in q:
        return f"{intro}\n\nFertilizer advice: {intelligence['fertilizer_text']} Nutrient condition is {nutrient_state}."
    if "risk" in q:
        return f"{intro}\n\nRisk is {report['risk']} because disease is {report['disease']} and crop health score is {report['health_score']}/100."
    return f"{intro}\n\nSmart recommendation: {', '.join(report['actions'])}. Weather is {report['weather']} and disease status is {report['disease']}."


def ensure_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []
    cleaned = []
    for item in st.session_state.chat_history:
        if isinstance(item, dict) and "question" in item and "answer" in item:
            cleaned.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            cleaned.append({"question": str(item[0]), "answer": str(item[1])})
    st.session_state.chat_history = cleaned


def line_chart(values, labels, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=values, mode="lines+markers", line=dict(color=color, width=3), marker=dict(size=8)))
    fig.update_layout(title=title, height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def bar_chart(values, labels, title, color):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=color))
    fig.update_layout(title=title, height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def gauge_chart(value, title, color, max_value=100):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title}, gauge={"axis": {"range": [0, max_value]}, "bar": {"color": color}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.title(APP_NAME)
    st.caption("Integrated Telangana smart farming platform with connected predictions, smart farm report, dashboards, and AI assistant.")

    with st.sidebar:
        st.header("Farm Setup")
        location = st.selectbox("Location", TELANGANA_DISTRICTS)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
        farm_type = st.selectbox("Select Farm Type", FARM_TYPES)
        auto_fill = st.checkbox("Use real farm simulation mode", value=True)
        defaults = SIMULATION_DEFAULTS[farm_type]

        if auto_fill:
            temperature_default = float(defaults["temperature"])
            humidity_default = float(defaults["humidity"])
            rainfall_default = float(defaults["rainfall"])
            moisture_default = float(defaults["soil_moisture"])
            ph_default = float(defaults["ph"])
            n_default = int(defaults["nitrogen"])
            p_default = int(defaults["phosphorus"])
            k_default = int(defaults["potassium"])
            ndvi_default = float(defaults["ndvi"])
        else:
            temperature_default = 29.0
            humidity_default = 65.0
            rainfall_default = 350.0
            moisture_default = 50.0
            ph_default = 6.8
            n_default = 55
            p_default = 28
            k_default = 30
            ndvi_default = 0.60

        soil_type = st.selectbox("Soil Type", SOIL_TYPES)
        area = st.number_input("Area (acres)", 0.5, 100.0, 2.0, 0.5)
        temperature = st.slider("Temperature (°C)", 10.0, 45.0, temperature_default, 0.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, humidity_default, 1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 1200.0, rainfall_default, 5.0)
        soil_moisture = st.slider("Soil Moisture (%)", 5.0, 100.0, moisture_default, 1.0)
        ph = st.slider("Soil pH", 4.0, 9.0, ph_default, 0.1)
        nitrogen = st.slider("Nitrogen (N)", 0, 140, n_default, 1)
        phosphorus = st.slider("Phosphorus (P)", 0, 120, p_default, 1)
        potassium = st.slider("Potassium (K)", 0, 140, k_default, 1)
        ndvi = st.slider("NDVI", 0.0, 1.0, ndvi_default, 0.01)

    top_recommendations = recommend_crop(season, soil_type, temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium)
    selected_crop = st.selectbox("Predicted / Selected Crop", [item["crop"] for item in top_recommendations])
    crop_confidence = next(item["confidence"] for item in top_recommendations if item["crop"] == selected_crop)
    yield_per_acre, yield_total, yield_confidence = predict_yield(selected_crop, area, rainfall, temperature, soil_moisture, nitrogen, phosphorus, potassium)
    disease_name, disease_confidence = predict_disease(selected_crop, ndvi, humidity, soil_moisture)
    nutrient_state = nutrient_health(nitrogen, phosphorus, potassium)
    weather_name = weather_label(rainfall, humidity)
    intelligence = build_intelligence_layer(selected_crop, disease_name, ndvi, soil_moisture, rainfall, temperature, nitrogen, phosphorus, potassium)
    risk_score, risk_level, health_score = calculate_risk_and_health(disease_name, ndvi, soil_moisture, nutrient_state)
    report = build_smart_farm_report(location, season, selected_crop, crop_confidence, yield_total, yield_confidence, weather_name, disease_name, disease_confidence, risk_level, health_score, intelligence)

    tabs = st.tabs(["🌾 Smart Farming", "👁 Crop Monitoring", "📊 Dashboard", "🤖 AI Assistant"])

    with tabs[0]:
        st.subheader("🌾 SMART FARM REPORT")
        st.markdown(f"""
### 🌾 SMART FARM REPORT
**Crop:** {report['crop']} ({report['crop_conf']}%)  
**Yield:** {report['yield_total']} tons ({report['yield_conf']}%)  
**Weather:** {report['weather']}  
**Disease:** {report['disease']} ({report['disease_conf']}%)  

**🔴 Risk Level:** {report['risk']}  
**💚 Health Score:** {report['health_score']}/100
""")
        st.markdown("**✅ Recommended Actions:**")
        for action in report["actions"]:
            st.write(f"- {action}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Crop Prediction", f"{report['crop']}", f"{report['crop_conf']}% confidence")
        m2.metric("Disease Prediction", report['disease'], f"{report['disease_conf']}% confidence")
        m3.metric("Risk Score", risk_score, report['risk'])
        m4.metric("Health Score", health_score)

        st.info(f"Intelligence Layer: irrigation = {intelligence['irrigation_level']}, fertilizer = {intelligence['fertilizer_level']}. Connected decision logic is active.")

    with tabs[1]:
        st.subheader("👁 Crop Monitoring")
        c1, c2 = st.columns(2)
        c1.write(f"NDVI Status: {ndvi_status(ndvi)}")
        c1.write(f"Weather Condition: {weather_name}")
        c1.write(f"Nutrient Health: {nutrient_state}")
        c2.write(f"Disease Status: {disease_name}")
        c2.write(f"Irrigation Plan: {intelligence['irrigation_text']}")
        c2.write(f"Fertilizer Plan: {intelligence['fertilizer_text']}")

        recommendation_table = []
        for item in top_recommendations:
            recommendation_table.append({
                "Crop": item["crop"],
                "Score": item["score"],
                "Confidence %": item["confidence"],
            })
        st.dataframe(recommendation_table, use_container_width=True)

    with tabs[2]:
        st.subheader("📊 Visual Dashboard")
        ndvi_series = [max(0.1, ndvi - 0.12), max(0.1, ndvi - 0.06), ndvi, min(1.0, ndvi + 0.04)]
        yield_series = [round(yield_total * 0.70, 2), round(yield_total * 0.84, 2), round(yield_total * 0.93, 2), yield_total]
        nutrient_values = [nitrogen, phosphorus, potassium]

        d1, d2 = st.columns(2)
        d1.plotly_chart(line_chart(ndvi_series, ["Week 1", "Week 2", "Week 3", "Week 4"], "📊 NDVI Graph", "green"), use_container_width=True)
        d2.plotly_chart(line_chart(yield_series, ["Stage 1", "Stage 2", "Stage 3", "Final"], "📈 Yield Prediction Graph", "royalblue"), use_container_width=True)

        d3, d4 = st.columns(2)
        d3.plotly_chart(bar_chart(nutrient_values, ["Nitrogen", "Phosphorus", "Potassium"], "📉 Soil Nutrients", ["#16a34a", "#f59e0b", "#2563eb"]), use_container_width=True)
        d4.plotly_chart(gauge_chart(health_score, "Farm Health Score", "#dc2626" if risk_level == "HIGH" else "#f59e0b" if risk_level == "MEDIUM" else "#16a34a"), use_container_width=True)

    with tabs[3]:
        st.subheader("🤖 AI Assistant")
        ensure_chat_history()
        user_q = st.text_input("Ask: What should I do? irrigation? fertilizer? risk? disease?")
        col1, col2 = st.columns(2)
        ask = col1.button("Ask AI")
        clear = col2.button("Clear Chat")

        if clear:
            st.session_state.chat_history = []

        if ask and user_q.strip():
            answer = build_chatbot_reply(user_q, report, intelligence, ndvi, nutrient_state)
            st.session_state.chat_history.append({"question": user_q, "answer": answer})

        for item in reversed(st.session_state.chat_history):
            st.markdown(f"**Farmer:** {item['question']}")
            st.markdown(item["answer"])
            st.markdown("---")

    st.markdown("### Added features")
    st.write("1. Smart Farm Report")
    st.write("2. Intelligence Layer connecting disease, irrigation, and fertilizer")
    st.write("3. Risk Score and Health Score")
    st.write("4. NDVI, Yield, and Soil Nutrient dashboard charts")
    st.write("5. Location and Season aware predictions")
    st.write("6. Real Farm Simulation Mode")
    st.write("7. Smarter AI assistant using model outputs as context")
    st.write("8. Confidence scores for crop, disease, and yield")


if __name__ == "__main__":
    main()
