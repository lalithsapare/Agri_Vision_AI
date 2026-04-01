import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Telangana AgriTech Assistant", page_icon="🌾", layout="wide")

TELANGANA_CROPS = {
    "Kharif": [
        "Cotton", "Maize", "Paddy", "Red Gram", "Soybean", "Green Gram",
        "Black Gram", "Jowar", "Bajra", "Sesame", "Groundnut", "Turmeric", "Chilli"
    ],
    "Rabi": [
        "Paddy", "Maize", "Groundnut", "Bengal Gram", "Black Gram", "Sesame",
        "Jowar", "Sunflower", "Safflower", "Horse Gram", "Castor"
    ],
    "Summer": [
        "Paddy", "Maize", "Groundnut", "Sesame", "Sunflower", "Vegetables", "Fodder Maize"
    ]
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

DEFAULTS = {
    "Kharif": {"temperature": 29.0, "humidity": 78.0, "rainfall": 710.0, "soil_moisture": 61.0, "ph": 6.8, "nitrogen": 72, "phosphorus": 38, "potassium": 41},
    "Rabi": {"temperature": 24.0, "humidity": 58.0, "rainfall": 95.0, "soil_moisture": 44.0, "ph": 7.0, "nitrogen": 55, "phosphorus": 30, "potassium": 35},
    "Summer": {"temperature": 34.0, "humidity": 42.0, "rainfall": 35.0, "soil_moisture": 29.0, "ph": 7.1, "nitrogen": 48, "phosphorus": 26, "potassium": 32},
}

CROP_GUIDE = {
    "Cotton": {"water": "Medium", "season": "Kharif", "soil": "Black Cotton Soil", "fert": "Balanced NPK with split nitrogen application"},
    "Maize": {"water": "Medium", "season": "Kharif/Rabi", "soil": "Red Sandy Loam", "fert": "Nitrogen rich basal + top dressing"},
    "Paddy": {"water": "High", "season": "Kharif/Rabi", "soil": "Clay Loam", "fert": "High nitrogen with zinc attention"},
    "Red Gram": {"water": "Low-Medium", "season": "Kharif", "soil": "Red Sandy Loam", "fert": "Starter phosphorus, low nitrogen"},
    "Soybean": {"water": "Medium", "season": "Kharif", "soil": "Alluvial Soil", "fert": "Rhizobium support + phosphorus"},
    "Green Gram": {"water": "Low", "season": "Kharif", "soil": "Red Sandy Loam", "fert": "Light phosphorus and sulphur"},
    "Black Gram": {"water": "Low", "season": "Kharif/Rabi", "soil": "Red Sandy Loam", "fert": "Low nitrogen, phosphorus support"},
    "Jowar": {"water": "Low", "season": "Kharif/Rabi", "soil": "Black Cotton Soil", "fert": "Moderate NPK with micronutrient watch"},
    "Bajra": {"water": "Low", "season": "Kharif", "soil": "Lateritic Soil", "fert": "Low input crop, moderate nitrogen"},
    "Sesame": {"water": "Low", "season": "Kharif/Rabi/Summer", "soil": "Red Sandy Loam", "fert": "Sulphur and phosphorus support"},
    "Groundnut": {"water": "Medium", "season": "Kharif/Rabi/Summer", "soil": "Red Sandy Loam", "fert": "Gypsum, calcium and phosphorus"},
    "Turmeric": {"water": "High", "season": "Kharif", "soil": "Clay Loam", "fert": "Organic matter rich with potassium support"},
    "Chilli": {"water": "Medium", "season": "Kharif", "soil": "Red Sandy Loam", "fert": "Potash and micronutrient support"},
    "Bengal Gram": {"water": "Low", "season": "Rabi", "soil": "Black Cotton Soil", "fert": "Phosphorus dominant recommendation"},
    "Sunflower": {"water": "Low-Medium", "season": "Rabi/Summer", "soil": "Alluvial Soil", "fert": "Sulphur and boron support"},
    "Safflower": {"water": "Low", "season": "Rabi", "soil": "Black Cotton Soil", "fert": "Low irrigation, balanced phosphorus"},
    "Horse Gram": {"water": "Low", "season": "Rabi", "soil": "Lateritic Soil", "fert": "Very low input, organic matter preferred"},
    "Castor": {"water": "Low-Medium", "season": "Rabi", "soil": "Black Cotton Soil", "fert": "Moderate NPK with sulphur"},
    "Vegetables": {"water": "Medium-High", "season": "Summer", "soil": "Clay Loam", "fert": "Crop-specific soluble nutrition"},
    "Fodder Maize": {"water": "Medium", "season": "Summer", "soil": "Red Sandy Loam", "fert": "Nitrogen rich fodder nutrition"},
}


def recommend_crop(season, soil_type, temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium):
    crops = TELANGANA_CROPS.get(season, [])
    scored = []
    for crop in crops:
        score = 50
        guide = CROP_GUIDE.get(crop, {})
        if guide.get("soil") == soil_type:
            score += 18
        if crop == "Cotton" and 24 <= temperature <= 32 and rainfall >= 500:
            score += 25
        if crop == "Paddy" and humidity >= 70 and rainfall >= 600:
            score += 25
        if crop == "Maize" and 20 <= temperature <= 32 and 400 <= rainfall <= 900:
            score += 22
        if crop == "Red Gram" and 22 <= temperature <= 34 and rainfall >= 450:
            score += 22
        if crop == "Groundnut" and 22 <= temperature <= 30 and 350 <= rainfall <= 700:
            score += 20
        if crop == "Bengal Gram" and season == "Rabi" and rainfall <= 150:
            score += 24
        if crop == "Sesame" and rainfall <= 500:
            score += 18
        if crop == "Jowar" and 24 <= temperature <= 35:
            score += 18
        if crop == "Turmeric" and humidity >= 75 and rainfall >= 800:
            score += 20
        if 6.0 <= ph <= 7.8:
            score += 8
        if nitrogen >= 50:
            score += 5
        if phosphorus >= 25:
            score += 5
        if potassium >= 25:
            score += 5
        scored.append((crop, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]


def estimate_yield(crop, area, rainfall, temperature, soil_moisture, nitrogen, phosphorus, potassium):
    base = {
        "Cotton": 18, "Maize": 32, "Paddy": 26, "Red Gram": 10, "Soybean": 14,
        "Green Gram": 8, "Black Gram": 8, "Jowar": 14, "Bajra": 12, "Sesame": 6,
        "Groundnut": 16, "Turmeric": 80, "Chilli": 22, "Bengal Gram": 10,
        "Sunflower": 9, "Safflower": 7, "Horse Gram": 6, "Castor": 11,
        "Vegetables": 95, "Fodder Maize": 140
    }.get(crop, 12)

    weather_factor = 1 + ((rainfall - 300) / 2000) + ((soil_moisture - 40) / 300)
    nutrient_factor = 1 + ((nitrogen + phosphorus + potassium) - 100) / 500
    temp_penalty = 1 - abs(28 - temperature) / 60
    final_per_acre = max(3, base * weather_factor * nutrient_factor * temp_penalty)
    total = round(final_per_acre * area, 2)
    return round(final_per_acre, 2), total


def irrigation_advice(crop, soil_moisture, temperature, rainfall):
    if soil_moisture < 30 or (temperature > 34 and rainfall < 50):
        status = "Immediate irrigation needed"
        advice = "Use drip or sprinkler if available; irrigate in evening or early morning to reduce evaporation loss."
    elif soil_moisture < 45:
        status = "Irrigation needed in 1-2 days"
        advice = "Plan light irrigation soon; avoid water stress during flowering and pod or boll formation."
    else:
        status = "Soil moisture currently acceptable"
        advice = "Monitor field moisture and avoid over-irrigation, especially in black soils and lowland zones."
    crop_note = CROP_GUIDE.get(crop, {}).get("water", "Medium")
    return status, advice, crop_note


def fertilizer_advice(crop, nitrogen, phosphorus, potassium):
    low = []
    if nitrogen < 50:
        low.append("Nitrogen")
    if phosphorus < 25:
        low.append("Phosphorus")
    if potassium < 25:
        low.append("Potassium")

    if not low:
        return f"NPK status looks acceptable for {crop}. Focus on split doses, micronutrients, and soil-test-based application."

    crop_specific = CROP_GUIDE.get(crop, {}).get("fert", "Balanced fertilization recommended")
    return f"Low nutrients detected: {', '.join(low)}. For {crop}, recommend: {crop_specific}."


def soil_health_status(ph, nitrogen, phosphorus, potassium):
    msgs = []
    if ph < 6.0:
        msgs.append("Soil is slightly acidic; liming may help in some fields.")
    elif ph > 8.0:
        msgs.append("Soil is alkaline; gypsum and organic matter may help improve nutrient availability.")
    else:
        msgs.append("Soil pH is in an acceptable range for many Telangana crops.")

    avg = (nitrogen + phosphorus + potassium) / 3
    if avg >= 50:
        msgs.append("Macronutrient condition is fairly good.")
    elif avg >= 30:
        msgs.append("Macronutrient condition is moderate; improve with balanced fertilization and FYM.")
    else:
        msgs.append("Macronutrient condition is low; soil-test-based nutrient correction is needed.")
    return " ".join(msgs)


def ndvi_status(value):
    if value >= 0.70:
        return "Healthy crop canopy", "Vegetation vigor is strong and crop stress appears low."
    if value >= 0.45:
        return "Moderate crop vigor", "Field is acceptable but should be checked for moisture, nutrient, or pest pressure."
    return "Low vegetation vigor", "Crop stress risk is high; inspect water, pests, disease, and nutrient deficiency immediately."


def build_agritech_reply(question, district, season, crop, soil_type, temp, humidity, rainfall, soil_moisture, ph, n, p, k, ndvi):
    q = question.lower().strip()
    ndvi_label, ndvi_note = ndvi_status(ndvi)
    irrig_status, irrig_note, water_need = irrigation_advice(crop, soil_moisture, temp, rainfall)
    fert_note = fertilizer_advice(crop, n, p, k)
    soil_note = soil_health_status(ph, n, p, k)

    intro = f"🌾 Artificial AgriTech Assistant: Telangana field guidance for {district}, {season} season, focused on {crop}."
    context = f"Current field context: temperature {temp}°C, humidity {humidity}%, rainfall {rainfall} mm, soil moisture {soil_moisture}%, pH {ph}, NPK {n}/{p}/{k}, NDVI {ndvi:.2f} ({ndvi_label})."

    if any(word in q for word in ["crop", "which crop", "recommend"]):
        top = recommend_crop(season, soil_type, temp, humidity, rainfall, ph, n, p, k)[:3]
        crop_line = ", ".join([f"{name} ({score} score)" for name, score in top])
        return f"{intro}\n\n{context}\n\nBest Telangana crop options now: {crop_line}. Soil type {soil_type} and season pattern are strongly supporting these choices."

    if any(word in q for word in ["water", "irrigation", "drip", "moisture"]):
        return f"{intro}\n\n{context}\n\nIrrigation decision: {irrig_status}. {irrig_note} Crop water need for {crop} is {water_need}."

    if any(word in q for word in ["fertilizer", "fertiliser", "npk", "urea", "dap", "potash"]):
        return f"{intro}\n\n{context}\n\nFertilizer advisory: {fert_note} Soil note: {soil_note}"

    if any(word in q for word in ["yield", "production", "output"]):
        per_acre, total = estimate_yield(crop, 1.0, rainfall, temp, soil_moisture, n, p, k)
        return f"{intro}\n\n{context}\n\nEstimated yield for {crop}: about {per_acre} units per acre under current conditions. Yield may reduce if stress continues, so follow irrigation and nutrient corrections quickly."

    if any(word in q for word in ["disease", "pest", "stress", "leaf", "yellow"]):
        return f"{intro}\n\n{context}\n\nStress interpretation: {ndvi_note} Also inspect leaves, stem base, and root zone for pests, sucking insects, wilt, or nutrient deficiency. Start with field scouting and remove severely affected plants where necessary."

    return f"{intro}\n\n{context}\n\nAgronomic advisory: {irrig_status}. {fert_note} {soil_note} NDVI note: {ndvi_note} For Telangana farming, take decisions crop stage wise and prefer soil-test-based correction instead of blanket application."


def draw_gauge(value, title, min_v, max_v, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_v, max_v]},
                "bar": {"color": color},
                "bgcolor": "white"
            }
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.title("🌾 Telangana Artificial AgriTech Assistant")
    st.caption("Region-focused farmer assistant for Telangana crops, irrigation, soil, NPK, NDVI, and advisory support.")

    with st.sidebar:
        st.header("Field Setup")
        district = st.selectbox("District", TELANGANA_DISTRICTS)
        season = st.selectbox("Season", list(TELANGANA_CROPS.keys()))
        defaults = DEFAULTS[season]
        soil_type = st.selectbox("Soil Type", SOIL_TYPES)
        crop = st.selectbox("Current Crop / Target Crop", TELANGANA_CROPS[season])
        area = st.number_input("Area (acres)", 0.5, 100.0, 2.0, 0.5)
        temperature = st.slider("Temperature (°C)", 10.0, 45.0, float(defaults["temperature"]), 0.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, float(defaults["humidity"]), 1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 1200.0, float(defaults["rainfall"]), 5.0)
        soil_moisture = st.slider("Soil Moisture (%)", 5.0, 100.0, float(defaults["soil_moisture"]), 1.0)
        ph = st.slider("Soil pH", 4.0, 9.5, float(defaults["ph"]), 0.1)
        nitrogen = st.slider("Nitrogen (N)", 0, 140, int(defaults["nitrogen"]), 1)
        phosphorus = st.slider("Phosphorus (P)", 0, 120, int(defaults["phosphorus"]), 1)
        potassium = st.slider("Potassium (K)", 0, 140, int(defaults["potassium"]), 1)
        ndvi = st.slider("NDVI", 0.0, 1.0, 0.62, 0.01)

    tabs = st.tabs([
        "Dashboard", "Crop Recommendation", "Yield", "Irrigation",
        "Fertilizer", "Soil & NPK", "NDVI", "Farmer Chat"
    ])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("District", district)
        c2.metric("Season", season)
        c3.metric("Crop", crop)
        c4.metric("Area", f"{area} acres")

        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(draw_gauge(soil_moisture, "Soil Moisture", 0, 100, "#0f766e"), use_container_width=True)
        g2.plotly_chart(draw_gauge(humidity, "Humidity", 0, 100, "#2563eb"), use_container_width=True)
        g3.plotly_chart(draw_gauge(ndvi, "NDVI", 0, 1, "#16a34a"), use_container_width=True)

        st.info(f"Telangana crop list for {season}: {', '.join(TELANGANA_CROPS[season])}")

    with tabs[1]:
        top_crops = recommend_crop(season, soil_type, temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium)
        st.subheader("Top Telangana crop recommendations")
        for idx, (name, score) in enumerate(top_crops, start=1):
            guide = CROP_GUIDE.get(name, {})
            st.write(
                f"{idx}. {name} — score {score} | Soil: {guide.get('soil', '-')} | "
                f"Water need: {guide.get('water', '-')} | Fertilizer note: {guide.get('fert', '-')}"
            )

    with tabs[2]:
        per_acre, total_yield = estimate_yield(crop, area, rainfall, temperature, soil_moisture, nitrogen, phosphorus, potassium)
        st.subheader("Yield estimation")
        st.metric("Estimated yield per acre", per_acre)
        st.metric("Estimated total yield", total_yield)
        st.write("This is a rule-based Telangana crop estimate useful for demo, planning, and project portfolios.")

    with tabs[3]:
        status, advice, water_need = irrigation_advice(crop, soil_moisture, temperature, rainfall)
        st.subheader("Irrigation advisory")
        st.success(status)
        st.write(advice)
        st.write(f"Crop water requirement category: {water_need}")

    with tabs[4]:
        st.subheader("Fertilizer recommendation")
        st.write(fertilizer_advice(crop, nitrogen, phosphorus, potassium))
        st.write(f"Crop-specific Telangana note: {CROP_GUIDE.get(crop, {}).get('fert', 'Balanced fertilization recommended')}")

    with tabs[5]:
        st.subheader("Soil and nutrient health")
        st.write(soil_health_status(ph, nitrogen, phosphorus, potassium))
        st.write(f"Selected soil type: {soil_type}")
        st.write(f"Current NPK values: {nitrogen}/{phosphorus}/{potassium}")

    with tabs[6]:
        label, note = ndvi_status(ndvi)
        st.subheader("NDVI and crop stress")
        st.write(f"Status: {label}")
        st.write(note)

    with tabs[7]:
        st.subheader("Farmer chat assistant")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_q = st.text_input("Ask in simple style: Which crop for my field? irrigation? fertilizer? yield? pest stress?")
        col_a, col_b = st.columns([1, 1])
        ask = col_a.button("Ask Assistant")
        clear = col_b.button("Clear Chat")

        if clear:
            st.session_state.chat_history = []

        if ask and user_q.strip():
            reply = build_agritech_reply(
                user_q, district, season, crop, soil_type, temperature, humidity,
                rainfall, soil_moisture, ph, nitrogen, phosphorus, potassium, ndvi
            )
            st.session_state.chat_history.append((user_q, reply))

        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**Farmer:** {q}")
            st.markdown(a)
            st.markdown("---")

    st.markdown("### Telangana crop focus used in this app")
    for s, crops in TELANGANA_CROPS.items():
        st.write(f"{s}: {', '.join(crops)}")


if __name__ == "__main__":
    main()
