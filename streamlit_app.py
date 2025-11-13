# streamlit_app.py  (TFLite-based, Python 3.12-ready)
from __future__ import annotations
import os
import pathlib
import requests
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Try to import tflite runtime; fallback to tensorflow.lite if available
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        from tensorflow.lite import Interpreter as TfLiteInterpreter  # type: ignore
        # create a thin wrapper so rest of the code uses same name
        class _TFWrapper:
            def __init__(self, model_path):
                self._interp = TfLiteInterpreter(model_path=model_path)
            def allocate_tensors(self):
                self._interp.allocate_tensors()
            def get_input_details(self):
                return self._interp.get_input_details()
            def get_output_details(self):
                return self._interp.get_output_details()
            def set_tensor(self, idx, val):
                self._interp.set_tensor(idx, val)
            def invoke(self):
                self._interp.invoke()
            def get_tensor(self, idx):
                return self._interp.get_tensor(idx)
        tflite = _TFWrapper  # type: ignore
    except Exception:
        tflite = None

st.set_page_config(page_title="Cotton Leaf Disease (TFLite)", layout="centered")
st.title("Cotton Leaf Disease Prediction — TFLite (Python 3.12)")
st.write("This app uses a TensorFlow Lite model for inference (no heavy TF install).")

# --- CONFIG: change if you host model elsewhere ---
TFLITE_PATH = "model_resnet.tflite"
TFLITE_URL = "https://raw.githubusercontent.com/DARK-art108/Cotton-Leaf-Disease-Prediction/main/model_resnet.tflite"
# If you host on GitHub Releases or HF, replace TFLITE_URL accordingly.

# CLASS MAPPING adjusted to match your model's indices (confirmed with sample)
CLASS_MAPPING = {
    0: "Fresh cotton leaf",
    1: "Fresh cotton plant",
    2: "Diseased cotton leaf",
    3: "Diseased cotton plant"
}

def download_tflite(model_path: str = TFLITE_PATH, url: str = TFLITE_URL, timeout: int = 60) -> str:
    p = pathlib.Path(model_path)
    if p.is_file():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner("Downloading TFLite model (may take a moment)..."):
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = str(p) + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, str(p))
        st.success("TFLite model downloaded.")
        return str(p)
    except Exception as e:
        st.error(f"Could not download TFLite model from {url} — {e}")
        raise

@st.cache_resource
def load_tflite_interpreter(model_path: str = TFLITE_PATH):
    if tflite is None:
        raise RuntimeError("No TFLite runtime available. Ensure tflite-runtime or tensorflow is installed.")
    # If tflite is the wrapper class, instantiate differently
    if hasattr(tflite, "Interpreter") or hasattr(tflite, "load_delegate"):
        # normal tflite_runtime module
        interpreter = tflite.Interpreter(model_path=model_path)
    else:
        # our wrapper class assigned earlier
        interpreter = tflite(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def prepare_image_for_tflite(pil_img: Image.Image, target_size=(224,224)):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    arr = (arr / 127.0) - 1.0
    batch = np.expand_dims(arr, axis=0)
    return batch

def predict_tflite(interpreter, input_details, output_details, pil_img: Image.Image):
    batch = prepare_image_for_tflite(pil_img)
    # Ensure dtype matches expected input dtype
    input_index = input_details[0]["index"]
    expected_dtype = input_details[0]["dtype"]
    # cast if needed
    batch_to_feed = batch.astype(expected_dtype)
    interpreter.set_tensor(input_index, batch_to_feed)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    label_idx = int(np.argmax(out, axis=1)[0])
    return label_idx, out

# Ensure model exists or instruct user how to provide it
try:
    model_file = download_tflite()
except Exception:
    st.error(
        "TFLite model missing. Please convert your HDF5 to TFLite locally (see instructions in the app README) "
        "and upload model_resnet.tflite to the app folder or provide a direct TFLite URL in TFLITE_URL."
    )
    st.stop()

# Load interpreter (cached)
try:
    interpreter, input_details, output_details = load_tflite_interpreter(model_file)
except Exception as e:
    st.error(f"Failed to load TFLite interpreter: {e}")
    st.stop()

# File uploader & UI
uploaded_file = st.file_uploader("Upload a cotton leaf image (jpg/png) ...", type=["jpg","jpeg","png"])
if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.spinner("Running TFLite inference..."):
            idx, scores = predict_tflite(interpreter, input_details, output_details, img)
        score_list = [float(x) for x in scores.flatten()]
        # Show scores table
        rows = [{"index": i, "score": score_list[i], "label": CLASS_MAPPING.get(i, f"Label {i}")} for i in range(len(score_list))]
        st.table(rows)
        st.success(f"Prediction: **{CLASS_MAPPING.get(idx, f'Label {idx}')}** (index {idx})")
        topk = sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)[:3]
        st.write("Top predictions (index, label, score):")
        st.write([(i, CLASS_MAPPING.get(i, f"Label {i}"), float(s)) for i, s in topk])
    except Exception as exc:
        st.error(f"Inference error: {exc}")
else:
    st.info("Upload an image to classify (or provide model_resnet.tflite in repo).")

st.markdown("---")
st.markdown(
    "If you don't have `model_resnet.tflite` yet, run the conversion script (see instructions below) on a machine "
    "that can load your original HDF5 model, then upload the generated `model_resnet.tflite` to this app's repo."
)
