import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="DisasterGuard AI", page_icon="🌍", layout="wide")

st.title("🌍 DisasterGuard AI")
st.subheader("AI-Powered Disaster Risk Prediction & Emergency Decision Support")
st.caption("Live Weather + ML Risk Analysis + Emergency Priority + Official Guidance")

@st.cache_resource
def load_ml_models():
    models = {
        "Flood": joblib.load("models/flood_model.pkl"),
        "Cyclone": joblib.load("models/cyclone_model.pkl"),
        "Landslide": joblib.load("models/landslide_model.pkl")
    }
    features = joblib.load("models/features.pkl")
    return models, features

try:
    ml_models, model_features = load_ml_models()
    models_loaded = True
except Exception as e:
    models_loaded, ml_models, model_features = False, {}, []
    st.warning(f"ML models could not be loaded: {e}")

@st.cache_data(ttl=600)
def get_city_data(city):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json", "countryCode": "IN"},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return data["results"][0] if data.get("results") else None
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_weather_data(latitude, longitude):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude, "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,cloud_cover,surface_pressure,wind_speed_10m,wind_gusts_10m",
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,precipitation_probability,wind_speed_10m,soil_moisture_0_to_7cm",
                "forecast_days": 2, "timezone": "auto"
            },
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def weather_description(code):
    codes = {
        0:"☀️ Clear Sky",1:"🌤️ Mainly Clear",2:"⛅ Partly Cloudy",3:"☁️ Overcast",
        45:"🌫️ Fog",48:"🌫️ Rime Fog",51:"🌦️ Light Drizzle",53:"🌦️ Moderate Drizzle",
        55:"🌧️ Dense Drizzle",61:"🌧️ Slight Rain",63:"🌧️ Moderate Rain",65:"🌧️ Heavy Rain",
        71:"❄️ Slight Snow",73:"❄️ Moderate Snow",75:"❄️ Heavy Snow",80:"🌦️ Rain Showers",
        81:"🌧️ Moderate Rain Showers",82:"⛈️ Violent Rain Showers",95:"⛈️ Thunderstorm",
        96:"⛈️ Thunderstorm + Hail",99:"⛈️ Severe Thunderstorm"
    }
    return codes.get(code, "🌍 Unknown Weather")

def get_risk_level(score):
    if score >= 80: return "CRITICAL", "🔴"
    if score >= 60: return "HIGH", "🟠"
    if score >= 35: return "MODERATE", "🟡"
    return "LOW", "🟢"

def predict_disaster_risk(values):
    X = pd.DataFrame([values])[model_features]
    return {
        name: float(np.clip(model.predict(X)[0], 0, 100))
        for name, model in ml_models.items()
    }

def calculate_priority(risk, population, vulnerability, road_access):
    population_factor = min((population / 100000) * 100, 100)
    return float(np.clip(
        0.55*risk + 0.20*population_factor +
        0.15*(vulnerability*100) + 0.10*(100-road_access), 0, 100
    ))

def explain_risk(disaster, rainfall, humidity, wind, soil, pressure, elevation):
    factors = []
    if disaster == "Flood":
        if rainfall > 50: factors.append(f"High rainfall ({rainfall:.1f} mm)")
        if humidity > 80: factors.append(f"High humidity ({humidity:.1f}%)")
        if soil > 0.5: factors.append("High soil moisture")
        if elevation < 150: factors.append("Low elevation")
    elif disaster == "Cyclone":
        if wind > 60: factors.append(f"Strong wind ({wind:.1f} km/h)")
        if pressure < 995: factors.append(f"Low atmospheric pressure ({pressure:.1f} hPa)")
        if humidity > 80: factors.append("High humidity")
    else:
        if rainfall > 50: factors.append(f"Heavy rainfall ({rainfall:.1f} mm)")
        if soil > 0.5: factors.append("High soil moisture")
        if elevation < 200: factors.append("Lower elevation")
    return factors or ["No major threshold exceeded"]

def get_official_guidance(disaster):
    data = {
        "Flood": [
            "Monitor official IMD rainfall and flood information.",
            "Prioritize low-lying and vulnerable areas.",
            "Keep emergency response resources ready.",
            "Avoid unnecessary travel through flooded areas.",
            "Follow evacuation instructions from authorized authorities."
        ],
        "Cyclone": [
            "Monitor official cyclone warnings.",
            "Keep emergency communication systems ready.",
            "Prioritize vulnerable populations.",
            "Keep emergency supplies ready.",
            "Follow official evacuation instructions when issued."
        ],
        "Landslide": [
            "Monitor heavy rainfall and slope conditions.",
            "Identify settlements near steep slopes.",
            "Monitor official disaster warnings.",
            "Prepare alternative safe routes.",
            "Follow instructions from authorized authorities."
        ]
    }
    return data.get(disaster, [])

st.sidebar.header("📍 Location")
city = st.sidebar.text_input("Enter City / District", placeholder="Example: Amravati")

st.sidebar.header("👥 Emergency Factors")
vulnerability = st.sidebar.slider("Community Vulnerability", 0.0, 1.0, 0.40, 0.05)
road_access = st.sidebar.slider("Road Accessibility (%)", 0, 100, 70)
previous_disaster = st.sidebar.selectbox("Previous Disaster History", ["No", "Yes"])

st.sidebar.header("🌍 Geographic Risk Factors")
river_level = st.sidebar.number_input("River Level (m)", 0.0, 10.0, 2.0, 0.1)
slope = st.sidebar.number_input("Slope (degrees)", 0.0, 60.0, 10.0, 1.0)
population_density = st.sidebar.number_input("Population Density", 1, 15000, 3000, 100)

analyze = st.sidebar.button("🚨 ANALYZE AREA", type="primary", use_container_width=True)

if analyze:
    if not city.strip():
        st.warning("Please enter a city or district.")
        st.stop()
    if not models_loaded:
        st.error("ML models are not loaded. Run train_model.py first.")
        st.stop()

    with st.spinner("Finding location and collecting weather data..."):
        location = get_city_data(city.strip())
    if location is None:
        st.error("Location not found. Please check the city name.")
        st.stop()

    lat, lon = location["latitude"], location["longitude"]
    city_name = location["name"]
    state = location.get("admin1", "Unknown")
    country = location.get("country", "India")
    elevation = location.get("elevation", 0) or 0
    population = location.get("population", 0) or population_density
    weather = get_weather_data(lat, lon)

    if weather is None:
        st.error("Unable to fetch weather data.")
        st.stop()

    current, hourly = weather["current"], weather["hourly"]
    temperature = current.get("temperature_2m", 0)
    humidity = current.get("relative_humidity_2m", 0)
    precipitation = current.get("precipitation", 0)
    pressure = current.get("surface_pressure", 1013)
    wind_speed = current.get("wind_speed_10m", 0)
    wind_gust = current.get("wind_gusts_10m", 0)
    code = current.get("weather_code", 0)
    soil_values = hourly.get("soil_moisture_0_to_7cm", [0])
    soil = soil_values[0] if soil_values else 0

    st.success(f"📍 {city_name}, {state}, {country}")

    st.subheader("📍 Location Intelligence")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Latitude", f"{lat:.4f}")
    c2.metric("Longitude", f"{lon:.4f}")
    c3.metric("Elevation", f"{elevation:.0f} m")
    c4.metric("Population", f"{population:,}")

    st.divider()
    st.subheader("🌦️ Live Environmental Conditions")
    st.info(weather_description(code))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🌡️ Temperature", f"{temperature} °C")
    c2.metric("💧 Humidity", f"{humidity} %")
    c3.metric("🌧️ Rainfall", f"{precipitation} mm")
    c4.metric("💨 Wind", f"{wind_speed} km/h")
    c1,c2,c3 = st.columns(3)
    c1.metric("📊 Pressure", f"{pressure} hPa")
    c2.metric("💨 Wind Gust", f"{wind_gust} km/h")
    c3.metric("🌱 Soil Moisture", f"{soil:.3f}")

    values = {
        "rainfall_mm": precipitation, "temperature_c": temperature,
        "humidity_pct": humidity, "river_level_m": river_level,
        "elevation_m": elevation, "soil_moisture_pct": soil,
        "wind_speed_kmh": wind_speed, "pressure_hpa": pressure,
        "slope_deg": slope, "population_density": population_density,
        "vulnerability_index": vulnerability, "road_access_index": road_access/100,
        "previous_event": 1 if previous_disaster == "Yes" else 0
    }

    st.divider()
    st.subheader("🤖 AI Multi-Disaster Risk Engine")
    results = predict_disaster_risk(values)
    cols = st.columns(3)
    for col, disaster in zip(cols, results):
        score = results[disaster]
        level, icon = get_risk_level(score)
        with col:
            st.metric(f"{icon} {disaster}", f"{score:.1f}/100")
            st.write(f"**{level} Risk**")
            st.progress(int(score))

    highest_disaster = max(results, key=results.get)
    highest_score = results[highest_disaster]
    level, icon = get_risk_level(highest_score)

    st.divider()
    st.subheader("🎯 Highest Risk")
    if level in ["CRITICAL","HIGH"]:
        st.error(f"{icon} {highest_disaster} — {highest_score:.1f}/100 — {level}")
    elif level == "MODERATE":
        st.warning(f"{icon} {highest_disaster} — {highest_score:.1f}/100 — {level}")
    else:
        st.success(f"{icon} {highest_disaster} — {highest_score:.1f}/100 — {level}")

    priority = calculate_priority(highest_score, population, vulnerability, road_access)
    priority_label = "🔴 CRITICAL" if priority >= 75 else "🟠 HIGH" if priority >= 55 else "🟡 MEDIUM" if priority >= 35 else "🟢 LOW"

    st.subheader("🚨 Emergency Response Priority")
    st.progress(int(priority))
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Disaster Risk", f"{highest_score:.1f}/100")
    p2.metric("Population Exposure", f"{population:,}")
    p3.metric("Priority Score", f"{priority:.1f}/100")
    p4.metric("Response Level", priority_label)

    st.markdown("### 🧠 Priority Factors")
    st.dataframe(pd.DataFrame({
        "Factor":["Disaster Risk","Population Exposure","Community Vulnerability","Road Accessibility"],
        "Value":[f"{highest_score:.1f}%",f"{population:,}",f"{vulnerability*100:.0f}%",f"{road_access}%"]
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 Why is this risk high?")
    for factor in explain_risk(highest_disaster, precipitation, humidity, wind_speed, soil, pressure, elevation):
        st.write("• " + factor)

    st.divider()
    st.subheader("📈 Next 24 Hours Forecast")
    forecast_df = pd.DataFrame({
        "Time":hourly["time"][:24],
        "Temperature":hourly["temperature_2m"][:24],
        "Rain":hourly["rain"][:24],
        "Rain Probability":hourly["precipitation_probability"][:24],
        "Wind":hourly["wind_speed_10m"][:24]
    })
    st.line_chart(forecast_df.set_index("Time")[["Temperature","Rain","Wind"]])

    
    st.divider()
    st.subheader("🏛️ Real / Official Data Layer")
    st.info("Official sources are provided for verification. DisasterGuard AI does not replace government warnings.")
    o1,o2,o3 = st.columns(3)
    with o1:
        st.markdown("### 🌧️ IMD Rainfall")
        st.write("District/state rainfall information and monitoring.")
        st.link_button("Open IMD Rainfall","https://mausam.imd.gov.in/responsive/rainfallinformation.php")
    with o2:
        st.markdown("### ⚠️ IMD Warning")
        st.write("Weather warnings and forecast information.")
        st.link_button("Open IMD Warning","https://mausam.imd.gov.in/responsive/districtWiseWarning.php?day=Day_1")
    with o3:
        st.markdown("### 🚨 NDMA SACHET")
        st.write("National disaster alerts and safety information.")
        st.link_button("Open SACHET","https://sachet.ndma.gov.in/")

    st.divider()
    st.subheader("📋 Official Guidance Layer")
    st.caption(f"Guidance for detected highest risk: **{highest_disaster}**")
    for item in get_official_guidance(highest_disaster):
        st.write("✅ " + item)
    st.link_button("📖 Open NDMA Dos & Don'ts","https://sachet.ndma.gov.in/DosDont")

    st.divider()
    st.subheader("🎯 Emergency Decision Summary")
    summary = pd.DataFrame({
        "Disaster":list(results.keys()),
        "Risk Score":[round(x,2) for x in results.values()],
        "Risk Level":[get_risk_level(x)[0] for x in results.values()]
    }).sort_values("Risk Score", ascending=False)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📥 Download Risk Report")
    report = pd.DataFrame({
        "Location":[city_name],"State":[state],"Latitude":[lat],"Longitude":[lon],
        "Temperature_C":[temperature],"Humidity_%":[humidity],"Rainfall_mm":[precipitation],
        "Wind_kmh":[wind_speed],"Pressure_hPa":[pressure],"Soil_Moisture":[soil],
        "Flood_Risk":[round(results["Flood"],2)],"Cyclone_Risk":[round(results["Cyclone"],2)],
        "Landslide_Risk":[round(results["Landslide"],2)],"Highest_Risk":[highest_disaster],
        "Emergency_Priority":[round(priority,2)],"Response_Level":[priority_label]
    })
    st.download_button(
        "📥 Download Risk Report",
        report.to_csv(index=False),
        file_name=f"{city_name}_disaster_report.csv",
        mime="text/csv"
    )

st.divider()
st.caption("⚠️ DisasterGuard AI is a hackathon decision-support prototype. Risk scores are model estimates, not official disaster warnings. Always verify current alerts with authorized government sources.")