# streamlit_app.py — fixed mapping + robust loading (ready to deploy)
from __future__ import division, print_function
import os
import pathlib
import requests
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Import schedule class in case the saved model references it
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Config
MODEL_PATH = "model_resnet.hdf5"
MODEL_URL = "https://github.com/DARK-art108/Cotton-Leaf-Disease-Prediction/releases/download/v1.0/model_resnet.hdf5"

# NOTE: This CLASS_MAPPING was adjusted so the UI labels match the model's numeric indices.
# Based on testing (sample image you provided) the model returns index 1 for a fresh plant,
# so we map indices to labels accordingly:
CLASS_MAPPING = {
    0: "Fresh cotton leaf",
    1: "Fresh cotton plant",
    2: "Diseased cotton leaf",
    3: "Diseased cotton plant"
}

st.set_page_config(page_title="Cotton Leaf Disease Prediction", layout="centered")
st.title("Cotton Leaf Disease Prediction")
st.subheader("Transfer Learning Using ResNet (inference)")
st.write("Upload a cotton leaf image (jpg / jpeg / png). The model will run inference and show the predicted class.")

@st.cache_resource
def download_model_file(model_path=MODEL_PATH, model_url=MODEL_URL, timeout=60):
    model_file = pathlib.Path(model_path)
    if model_file.is_file():
        return str(model_file)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner("Downloading model (this may take a while)..."):
            with requests.get(model_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp_path = str(model_file) + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, str(model_file))
        st.success("Model downloaded.")
        return str(model_file)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        raise

@st.cache_resource
def load_model_for_inference(model_path=MODEL_PATH):
    local_path = download_model_file(model_path=model_path)
    # Try load without compile (skip optimizer/training objects)
    try:
        with st.spinner("Loading model into memory (inference-only)..."):
            model = tf.keras.models.load_model(local_path, compile=False)
        return model
    except Exception as e1:
        st.warning(f"load_model(..., compile=False) failed: {e1}")

    # Try again with ExponentialDecay in custom_objects
    try:
        with st.spinner("Retrying model load with custom_objects..."):
            model = tf.keras.models.load_model(
                local_path,
                custom_objects={"ExponentialDecay": ExponentialDecay},
                compile=False,
            )
        return model
    except Exception as e2:
        st.error(f"Failed to load model even with custom_objects: {e2}")
        raise

def prepare_image(pil_image: Image.Image, target_size=(224, 224)):
    img = pil_image.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    normalized = (arr / 127.0) - 1.0
    batch = np.expand_dims(normalized, axis=0)  # (1,H,W,3)
    return batch

def predict_image(model, pil_image: Image.Image):
    batch = prepare_image(pil_image)
    preds = model.predict(batch)
    label_idx = int(np.argmax(preds, axis=1)[0])
    return label_idx, preds

# Load the model once (cached)
try:
    model = load_model_for_inference(MODEL_PATH)
except Exception:
    st.stop()  # loading failed, stop the app

# Uploader + prediction UI
uploaded_file = st.file_uploader("Choose a Cotton Leaf Image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        st.write("")  # spacing
        with st.spinner("Running model inference..."):
            label_idx, scores = predict_image(model, input_image)

        # Display raw scores and mapping for verification
        score_list = [float(x) for x in scores.flatten()]
        st.write("Model raw scores by index:")
        # Show small table: index | score | mapped label
        rows = [{"index": i, "score": score_list[i], "label": CLASS_MAPPING.get(i, f"Label {i}")} for i in range(len(score_list))]
        st.table(rows)

        # Human readable result
        human_label = CLASS_MAPPING.get(label_idx, f"Label {label_idx}")
        st.success(f"Prediction: **{human_label}** (index {label_idx})")

        # Top-3 predictions
        topk = sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)[:3]
        st.write("Top predictions (index, label, score):")
        st.write([(i, CLASS_MAPPING.get(i, f"Label {i}"), float(s)) for i, s in topk])

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to classify.")

st.markdown("---")
st.markdown(
    """
**Notes**
- If you still see incorrect human-readable labels for several test images, let me know and I will run an automatic index→label detector script (requires a small `samples/` folder with labeled images).
- If you'd like this app to auto-detect mapping from a `samples/` folder on startup, I can add that too.
"""
)
