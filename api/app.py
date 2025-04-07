import os
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import folium
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Helper: Read Docker secret (returns path for binary files)
def read_secret(secret_file, binary=False):
    path = f"/run/secrets/{secret_file}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Secret file '{secret_file}' not found.")
    
    if binary:
        with open(path, "r") as f:
            return f.read().strip()  # Path to .keras
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

# 🔐 Load secrets (model_path and road_path point to actual files in container)
model_path = read_secret("model_path", binary=True)
road_path = read_secret("road_path", binary=True)

# ✅ Load model and road network
model = tf.keras.models.load_model(model_path, compile=False)

with open(road_path, "rb") as f:
    road_network = pickle.load(f)

# 🔍 Nearest node finder
def find_nearest_node(lat, lon):
    return min(road_network.nodes, key=lambda node: np.sqrt((lat - node[0])**2 + (lon - node[1])**2))

# 🔥 Wildfire Prediction Endpoint
@app.route('/predict_wildfire', methods=['POST'])
def predict_wildfire():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        if df.shape != (4096, 12):
            return jsonify({"error": f"Expected shape (4096, 12), got {df.shape}"}), 400

        coords = df[["latitude", "longitude"]].values
        input_data = df.to_numpy().reshape(1, 4096, 12)
        predictions = model.predict(input_data).reshape(-1)

        risk_score = float(np.mean(predictions))
        risk_level = "High" if np.max(predictions) > 0.6 else "Low" if risk_score <= 0.4 else "Medium"
        top_indices = predictions.argsort()[-15:][::-1]
        fire_zones = coords[top_indices].tolist()

        return jsonify({
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fire_zones": fire_zones
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🗺️ Evacuation Map Generator
@app.route('/generate_evacuation_map', methods=['POST'])
def generate_evacuation_map():
    try:
        data = request.json
        fire_zones = data.get("fire_zones", [])
        user_lat = data.get("user_latitude")
        user_lon = data.get("user_longitude")

        if not fire_zones:
            return jsonify({"error": "No fire zones provided"}), 400

        fire_coords = [tuple(coord) for coord in fire_zones[:10]]
        fire_center = np.mean(np.array(fire_coords), axis=0)
        all_nodes = [tuple(node) for node in road_network.nodes]

        safe_zones = sorted(
            all_nodes,
            key=lambda node: np.linalg.norm(np.array(node) - fire_center),
            reverse=True
        )[:5]

        rng = np.random.default_rng(seed=42)
        sampled_nodes = rng.choice(all_nodes, size=1000, replace=False)
        blue_nodes = []
        for node in sampled_nodes:
            if tuple(node) not in fire_coords and tuple(node) not in safe_zones:
                blue_nodes.append(tuple(node))
            if len(blue_nodes) == 20:
                break

        evac_map = folium.Map(location=fire_center, zoom_start=13)

        for coord in fire_coords:
            folium.CircleMarker(coord, radius=6, color='red', fill=True, fill_opacity=0.8).add_to(evac_map)

        for node in blue_nodes:
            folium.Marker(node, icon=folium.Icon(color='blue', icon='info-sign')).add_to(evac_map)

        for i, node in enumerate(safe_zones):
            folium.Marker(location=node, popup=f"Safe Zone {i+1}", icon=folium.Icon(color='green', icon='ok-sign')).add_to(evac_map)

        if user_lat is not None and user_lon is not None:
            folium.Marker([user_lat, user_lon], popup="Your Location", icon=folium.Icon(color='black', icon='star')).add_to(evac_map)

        map_html = evac_map.get_root().render()

        return jsonify({
            "map_html": map_html,
            "destinations": [{"latitude": s[0], "longitude": s[1]} for s in safe_zones]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🚀 Run
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
