import os
import streamlit as st
import requests
import streamlit.components.v1 as components

st.set_page_config(page_title="🔥 Wildfire Risk & Evacuation", layout="centered")
st.title("🔥 Wildfire Spread Prediction and Evacuation Planner")

# ✅ Load API URL from Docker secret
def read_api_url_secret():
    secret_path = "/run/secrets/api_url"
    if os.path.exists(secret_path):
        with open(secret_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return os.getenv("API_URL", "http://wildfire-api:5000")

API_URL = read_api_url_secret()

# 🔼 Upload CSV
st.header("📤 Upload Wildfire Data (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file (4096 rows × 12 features)", type=["csv"])

# 📍 Optional Coordinates
st.header("📍 Enter Your Location (Optional)")
default_lat = 37.7749
default_lon = -122.4194
user_lat = st.number_input("Your Latitude", value=default_lat, format="%.6f")
user_lon = st.number_input("Your Longitude", value=default_lon, format="%.6f")

# 🔥 Step 1: Predict Wildfire Zones
if uploaded_file:
    st.success("✅ File uploaded successfully!")

    if st.button("🚨 Predict Wildfire Zones"):
        with st.spinner("Predicting..."):
            uploaded_file.seek(0)
            response = requests.post(
                f"{API_URL}/predict_wildfire",
                files={"file": uploaded_file}
            )

        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                fire_zones = result["fire_zones"]
                st.session_state.fire_zones = fire_zones

                st.subheader("🔥 Wildfire Risk Prediction")
                st.markdown(f"**Risk Level:** `{result['risk_level']}`")
                st.markdown(f"**Risk Score:** `{result['risk_score']:.4f}`")
        else:
            st.error("Prediction failed. Please check your file.")
            st.code(response.text)

# 🗺️ Step 2: Generate Map
if "fire_zones" in st.session_state and st.session_state.fire_zones:
    if st.button("🗺️ Generate Evacuation Map"):
        with st.spinner("Building evacuation map..."):
            try:
                map_response = requests.post(
                    f"{API_URL}/generate_evacuation_map",
                    json={
                        "fire_zones": st.session_state.fire_zones,
                        "user_latitude": user_lat,
                        "user_longitude": user_lon
                    }
                )

                if map_response.status_code == 200:
                    map_data = map_response.json()
                    st.subheader("🗺️ Evacuation Route Map")
                    components.html(map_data["map_html"], height=600)

                    st.subheader("✅ Suggested Safe Zones")
                    for i, dest in enumerate(map_data["destinations"], start=1):
                        st.markdown(
                            f"🟢 Safe Zone {i}: Latitude: `{dest['latitude']:.5f}` | Longitude: `{dest['longitude']:.5f}`"
                        )
                else:
                    st.error("❌ Failed to generate evacuation map.")
                    st.markdown("### 🔍 Server Response:")
                    st.code(map_response.text)

            except Exception as e:
                st.error("❌ Error connecting to backend.")
                st.exception(e)
