# streamlit_app.py  — cleaned & deploy-friendly
from __future__ import division, print_function
import os
import pathlib
import requests
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

MODEL_PATH = "model_resnet.hdf5"
MODEL_URL = "https://github.com/DARK-art108/Cotton-Leaf-Disease-Prediction/releases/download/v1.0/model_resnet.hdf5"

st.title("Cotton Leaf Disease Prediction")
st.header("Transfer Learning Using ResNet")
st.text("Upload a Cotton Leaf Image (jpg, jpeg, png)")

@st.cache_resource
def download_and_load_model(model_path=MODEL_PATH, model_url=MODEL_URL):
    # ensure local file exists (download once if missing)
    if not pathlib.Path(model_path).is_file():
        st.info("Model not found locally — downloading model (this may take a while).")
        try:
            # stream download so we don't load whole file in memory
            with requests.get(model_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            raise

    # load model once (cached)
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise
    return model

model = download_and_load_model()

def model_predict(img: Image.Image, model):
    # prepare image
    size = (224, 224)
    img = ImageOps.fit(img.convert("RGB"), size, Image.LANCZOS)
    img_array = np.asarray(img).astype(np.float32)
    normalized = (img_array / 127.0) - 1.0
    data = np.expand_dims(normalized, axis=0)  # shape (1,224,224,3)
    preds = model.predict(data)
    label = int(np.argmax(preds, axis=1)[0])  # scalar label 0..3
    return label, preds

uploaded_file = st.file_uploader("Choose a Cotton Leaf Image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Cotton Leaf Image", use_column_width=True)
        st.write("Classifying... (model inference may take a few seconds)")
        label, preds = model_predict(img, model)
        # map label to human-readable result (adjust as per your class mapping)
        mapping = {
            0: "Diseased cotton leaf",
            1: "Diseased cotton plant",
            2: "Fresh cotton leaf",
            3: "Fresh cotton plant"
        }
        st.success(f"Prediction: {mapping.get(label, 'Unknown')}")
        st.write("Raw model scores:", preds.tolist())
    except Exception as e:
        st.error(f"Error during prediction: {e}")
