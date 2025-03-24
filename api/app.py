import os
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import folium
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Load model
model_path = os.path.join(os.path.dirname(__file__), "../models/wildfire_model.keras")
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ Load road network
road_network_path = os.path.join(os.path.dirname(__file__), "../data-collection/processed/road_network.gpickle")
with open(road_network_path, "rb") as f:
    road_network = pickle.load(f)

# ✅ Helper: closest node
def find_nearest_node(lat, lon):
    return min(road_network.nodes, key=lambda node: np.sqrt((lat - node[0])**2 + (lon - node[1])**2))

# 🔹 Predict wildfire
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
        if np.max(predictions) > 0.6:
            risk_level = "High"
        elif risk_score <= 0.4:
            risk_level = "Low"
        else:
            risk_level = "Medium"

        top_indices = predictions.argsort()[-15:][::-1]
        fire_zones = coords[top_indices].tolist()

        return jsonify({
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fire_zones": fire_zones
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Generate map
@app.route('/generate_evacuation_map', methods=['POST'])
def generate_evacuation_map():
    try:
        data = request.json
        fire_zones = data.get("fire_zones", [])
        user_lat = data.get("user_latitude")
        user_lon = data.get("user_longitude")

        if not fire_zones:
            return jsonify({"error": "No fire zones provided"}), 400

        # ✅ Ensure all coords are tuples
        fire_coords = [tuple(coord) for coord in fire_zones[:10]]
        fire_center = np.mean(np.array(fire_coords), axis=0)
        all_nodes = [tuple(node) for node in road_network.nodes]

        # 🟢 5 farthest safe zones
        safe_zones = sorted(
            all_nodes,
            key=lambda node: np.linalg.norm(np.array(node) - fire_center),
            reverse=True
        )[:5]

        # 🔵 20 safe-ish random evac points
        rng = np.random.default_rng(seed=42)
        sampled_nodes = rng.choice(all_nodes, size=1000, replace=False)
        blue_nodes = []
        for node in sampled_nodes:
            if tuple(node) not in fire_coords and tuple(node) not in safe_zones:
                blue_nodes.append(tuple(node))
            if len(blue_nodes) == 20:
                break

        # 🌍 Build map
        evac_map = folium.Map(location=fire_center, zoom_start=13)

        for coord in fire_coords:
            folium.CircleMarker(coord, radius=6, color='red', fill=True, fill_opacity=0.8).add_to(evac_map)

        for node in blue_nodes:
            folium.Marker(node, icon=folium.Icon(color='blue', icon='info-sign')).add_to(evac_map)

        for i, node in enumerate(safe_zones):
            folium.Marker(
                location=node,
                popup=f"Safe Zone {i+1}",
                icon=folium.Icon(color='green', icon='ok-sign')
            ).add_to(evac_map)

        if user_lat is not None and user_lon is not None:
            folium.Marker(
                [user_lat, user_lon],
                popup="Your Location",
                icon=folium.Icon(color='black', icon='star')
            ).add_to(evac_map)

        map_html = evac_map.get_root().render()
        return jsonify({
            "map_html": map_html,
            "destinations": [{"latitude": s[0], "longitude": s[1]} for s in safe_zones]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
