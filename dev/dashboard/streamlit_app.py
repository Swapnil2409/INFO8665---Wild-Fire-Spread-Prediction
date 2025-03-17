import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import folium_static

st.title("🔥 Wildfire Prediction & Evacuation Planning")

# ✅ Upload CSV File
st.sidebar.header("Upload Wildfire Data CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    st.sidebar.success("File uploaded successfully!")

# ✅ Predict Wildfire Risk
if st.sidebar.button("Predict Wildfire Risk"):
    if uploaded_file:
        files = {"file": uploaded_file}
        response = requests.post("http://127.0.0.1:5000/predict_wildfire", files=files)
        result = response.json()

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.success(f"🔥 Risk Level: {result['risk_level']} (Score: {result['risk_score']:.2f})")
    else:
        st.warning("Please upload a CSV file first.")

# ✅ Compute Evacuation Route
latitude = st.sidebar.number_input("Latitude", value=37.77)
longitude = st.sidebar.number_input("Longitude", value=-122.42)

if st.sidebar.button("Compute Evacuation Route"):
    data = {"latitude": latitude, "longitude": longitude}
    response = requests.post("http://127.0.0.1:5000/compute_routes", json=data)
    result = response.json()

    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        st.success("✅ Evacuation Route Computed!")
        st.json(result["route"])

# ✅ Generate Evacuation Map
if st.sidebar.button("Generate Evacuation Map"):
    data = {"latitude": latitude, "longitude": longitude}
    response = requests.post("http://127.0.0.1:5000/generate_evacuation_map", json=data)
    result = response.json()

    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        st.success("✅ Evacuation Map Generated!")
        evac_map = folium.Map(location=[latitude, longitude], zoom_start=12)
        folium.Marker([latitude, longitude], popup="Evacuation Point", icon=folium.Icon(color="blue")).add_to(evac_map)
        folium_static(evac_map)
