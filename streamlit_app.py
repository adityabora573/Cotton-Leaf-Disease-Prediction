# streamlit_app.py — fixed, deploy-friendly version
from __future__ import division, print_function
import os
import pathlib
import requests
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Optional: explicitly import ExponentialDecay in case model's optimizer used it
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Configuration
MODEL_PATH = "model_resnet.hdf5"
MODEL_URL = "https://github.com/DARK-art108/Cotton-Leaf-Disease-Prediction/releases/download/v1.0/model_resnet.hdf5"

# UI header
st.set_page_config(page_title="Cotton Leaf Disease Prediction", layout="centered")
st.title("Cotton Leaf Disease Prediction")
st.subheader("Transfer Learning Using ResNet (inference)")
st.write("Upload a cotton leaf image (jpg / jpeg / png). The model will run inference and show the predicted class.")

@st.cache_resource
def download_model_file(model_path=MODEL_PATH, model_url=MODEL_URL, timeout=60):
    """
    Ensure model file exists locally. If missing, download it from model_url (streaming).
    Returns the local model_path.
    """
    model_file = pathlib.Path(model_path)
    if model_file.is_file():
        return str(model_file)

    # Create parent directory if needed
    model_file.parent.mkdir(parents=True, exist_ok=True)

    # Stream-download with requests
    try:
        with st.spinner("Downloading model (this may take a while)..."):
            with requests.get(model_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # Write to temporary file first to avoid partial corruption
                tmp_path = str(model_file) + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                # Move/rename to final path
                os.replace(tmp_path, str(model_file))
        st.success("Model downloaded.")
        return str(model_file)
    except Exception as e:
        # Propagate an informative error
        st.error(f"Failed to download model: {e}")
        raise

@st.cache_resource
def load_model_for_inference(model_path=MODEL_PATH):
    """
    Load the Keras model for inference only (skip optimizer/training config).
    Tries multiple safe ways:
      1) load_model(..., compile=False)
      2) load_model(..., compile=False, custom_objects={ExponentialDecay: ...})
    Returns loaded model.
    """
    # ensure file exists (download if necessary)
    local_path = download_model_file(model_path=model_path)

    # Attempt 1: load without compile (skip optimizer)
    try:
        with st.spinner("Loading model into memory (inference-only)..."):
            model = tf.keras.models.load_model(local_path, compile=False)
        return model
    except Exception as e1:
        st.warning(f"load_model(..., compile=False) failed: {e1}")

    # Attempt 2: try again with known schedule in custom_objects
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

def prepare_image_for_model(pil_image: Image.Image, target_size=(224, 224)):
    """
    Convert PIL image to model-ready float32 numpy array of shape (1, H, W, 3),
    normalized to (x/127.0) - 1 as in the original code.
    """
    img = pil_image.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    normalized = (arr / 127.0) - 1.0
    batch = np.expand_dims(normalized, axis=0)  # (1, H, W, 3)
    return batch

def predict_image(model, pil_image: Image.Image):
    """
    Run inference on the model and return (label_index:int, scores:np.array).
    """
    batch = prepare_image_for_model(pil_image)
    preds = model.predict(batch)
    # preds shape (1, num_classes)
    label_idx = int(np.argmax(preds, axis=1)[0])
    return label_idx, preds

# Load model once (cached)
try:
    model = load_model_for_inference(MODEL_PATH)
except Exception:
    st.stop()  # model couldn't be loaded; stop further execution

# Class mapping — adjust if your model uses a different ordering
CLASS_MAPPING = {
    0: "Diseased cotton leaf",
    1: "Diseased cotton plant",
    2: "Fresh cotton leaf",
    3: "Fresh cotton plant"
}

# File uploader and prediction UI
uploaded_file = st.file_uploader("Choose a Cotton Leaf Image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        st.write("")  # spacing
        with st.spinner("Running model inference..."):
            label_idx, scores = predict_image(model, input_image)
        human_label = CLASS_MAPPING.get(label_idx, f"Label {label_idx}")
        st.success(f"Prediction: **{human_label}**")
        st.write("Model scores (raw outputs):")
        # show as list for readability
        st.write([float(x) for x in scores.flatten()])
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to classify.")

# Optional: small footer with troubleshooting tips
st.markdown("---")
st.markdown(
    """
**Notes & troubleshooting**
- The app loads a pre-trained model file from a GitHub Release. First run may take some time while the model downloads.
- If model loading fails due to Keras/TensorFlow version mismatches, consider re-saving the model without optimizer (include_optimizer=False) in an environment where it loads successfully, then re-upload that artifact.
"""
)
