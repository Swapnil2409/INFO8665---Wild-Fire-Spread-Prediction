import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import pickle
import folium
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Load Wildfire Prediction Model
model_path = os.path.join(os.path.dirname(__file__), "../models/wildfire_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ ML Model Loaded Successfully!")

# ✅ Load Road Network Graph
road_network_path = os.path.join(os.path.dirname(__file__), "../data-collection/processed/road_network.gpickle")
if not os.path.exists(road_network_path):
    raise FileNotFoundError(f"Road Network file not found: {road_network_path}")

with open(road_network_path, "rb") as f:
    road_network = pickle.load(f)
print("✅ Road Network Loaded Successfully!")

# ✅ Find the nearest node in the graph
def find_nearest_node(lat, lon):
    min_distance = float("inf")
    nearest_node = None
    for node in road_network.nodes:
        node_lat, node_lon = node
        distance = np.sqrt((lat - node_lat) ** 2 + (lon - node_lon) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node

# ------------------------------------
# 🔹 API Endpoint: Predict Wildfire Risk
# ------------------------------------
@app.route('/predict_wildfire', methods=['POST'])
def predict_wildfire():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # ✅ Ensure correct shape (4096, 12)
        if df.shape != (4096, 12):
            return jsonify({"error": f"Invalid input format. Expected (4096,12), got {df.shape}"}), 400

        input_data = df.to_numpy().reshape(1, 4096, 12)

        # ✅ Make prediction
        prediction = model.predict(input_data)

        # ✅ Compute risk level
        risk_score = np.mean(prediction)
        risk_level = "Low" if risk_score <= 0.4 else "Medium" if risk_score <= 0.7 else "High"

        return jsonify({
            "message": "Wildfire risk predicted",
            "risk_score": float(risk_score),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------
# 🔹 API Endpoint: Compute Evacuation Routes
# ------------------------------------
@app.route('/compute_routes', methods=['POST'])
def compute_routes():
    try:
        data = request.json
        lat, lon = data.get("latitude"), data.get("longitude")

        if lat is None or lon is None:
            return jsonify({"error": "Missing latitude or longitude"}), 400

        start = find_nearest_node(lat, lon)
        if start is None:
            return jsonify({"error": "Invalid location"}), 400

        # ✅ Find the safest destination dynamically
        all_nodes = list(road_network.nodes)
        destination = min(all_nodes, key=lambda x: nx.shortest_path_length(road_network, source=start, target=x, weight="risk"))

        # ✅ Compute shortest route
        route = nx.shortest_path(road_network, source=start, target=destination, weight="risk")

        return jsonify({"message": "Evacuation route computed", "route": route})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------
# 🔹 API Endpoint: Generate Evacuation Map
# ------------------------------------
@app.route('/generate_evacuation_map', methods=['POST'])
def generate_evacuation_map():
    try:
        data = request.json
        lat, lon = data.get("latitude"), data.get("longitude")

        if lat is None or lon is None:
            return jsonify({"error": "Missing latitude or longitude"}), 400

        start = find_nearest_node(lat, lon)
        if start is None:
            return jsonify({"error": "Invalid location"}), 400

        # ✅ Find the safest destination dynamically
        all_nodes = list(road_network.nodes)
        destination = min(all_nodes, key=lambda x: nx.shortest_path_length(road_network, source=start, target=x, weight="risk"))

        # ✅ Compute route
        route = nx.shortest_path(road_network, source=start, target=destination, weight="risk")

        # ✅ Create map
        evac_map = folium.Map(location=[lat, lon], zoom_start=12)

        # ✅ Add route markers
        for point in route:
            folium.Marker([point[0], point[1]], popup="Route Point", icon=folium.Icon(color="blue")).add_to(evac_map)

        # ✅ Save map
        map_path = os.path.join(os.path.dirname(__file__), "../dev/dashboard/static/evacuation_map.html")
        evac_map.save(map_path)

        return jsonify({
            "message": "Evacuation map generated",
            "map_url": "/static/evacuation_map.html"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
