# app.py - Flask API for Wildfire Spread Prediction

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "../models/wildfire_model.keras"  # Adjust path if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Define input feature names (Must match the training input features)
INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Wildfire Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input, expecting JSON payload"}), 400

        # Extract features from request and ensure correct order
        features = np.array([[data[feature] for feature in INPUT_FEATURES]], dtype=np.float32)

        # Make prediction
        prediction = model.predict(features)

        # Convert prediction to probability
        prediction_prob = prediction.tolist()

        return jsonify({"prediction": prediction_prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
