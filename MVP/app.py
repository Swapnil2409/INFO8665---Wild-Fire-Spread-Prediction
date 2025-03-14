# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
import numpy as np
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/wildfire_model.keras")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🔥 Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


MODEL_PATH = os.path.join(os.getcwd(), "models", "wildfire_model.keras")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🔥 Model file not found: {MODEL_PATH}")


model = tf.keras.models.load_model(MODEL_PATH)

INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']


st.title("🔥 Wildfire Prediction App")
st.write("Input environmental characteristics to predict the probability of fire occurrence")


input_data = {}
for feature in INPUT_FEATURES:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)


if st.button("Predict Wildfire Risk"):
    try:
        # 组织输入数据
        features = np.array([[input_data[feature] for feature in INPUT_FEATURES]], dtype=np.float32)

        # 🚀 修正形状，让它符合模型输入 (1, 4096, 12)
        features = np.tile(features, (1, 4096, 1))  # 复制 4096 次，使其变成 (1, 4096, 12)

        # 打印调试
        print("🔍 Adjusted input shape:", features.shape)  # 应该输出 (1, 4096, 12)

        # 进行预测
        prediction = model.predict(features)

        # 显示预测结果
        st.success(f"🔥 Predicting the probability of fire occurrence: {float(prediction[0][0]):.4f}")


    except Exception as e:
        st.error(f"error: {e}")
    