import streamlit as st
import torch
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
from huggingface_hub import hf_hub_download
from src.model import load_model
from src.gradcam_viz import get_gradcam
import config, tempfile

os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

if not os.path.exists("outputs/checkpoints/best_model.pth"):
    with st.spinner("Downloading model weights..."):
        hf_hub_download(
            repo_id="Sayanjib/leaf-disease-classifier",
            filename="best_model.pth",
            local_dir="outputs/checkpoints"
        )

if not os.path.exists("models/class_names.json"):
    with st.spinner("Downloading class names..."):
        hf_hub_download(
            repo_id="Sayanjib/leaf-disease-classifier",
            filename="class_names.json",
            local_dir="models"
        )

st.set_page_config(page_title="🌿 Leaf Disease Classifier", layout="wide")
st.title("🌿 Leaf Disease Classifier")
st.markdown("Upload a leaf image to classify the disease and visualize what the model focuses on.")

@st.cache_resource
def load():
    with open("models/class_names.json") as f:
        class_names = json.load(f)
    model = load_model(len(class_names), "outputs/checkpoints/best_model.pth")
    model.eval()
    return model, class_names

model, class_names = load()

uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Classifying..."):
        pred_class, pred_conf = get_gradcam(model, tmp_path, class_names)

    with col2:
        st.image(f"{config.PLOTS_DIR}/gradcam_{os.path.basename(tmp_path)}",
                 caption="GradCAM Heatmap", use_container_width=True)

    st.success(f"**Predicted:** `{pred_class}`  |  **Confidence:** `{pred_conf:.2%}`")