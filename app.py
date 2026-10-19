"""
Plant Disease Detection - Streamlit web app (live demo)
Supports: image upload + live camera capture
Run:  streamlit run app.py
"""

import json
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a live photo, and the model will predict the disease.")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("outputs/plant_model.keras")
    with open("outputs/class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def clean_name(name):
    """Make folder-style class names readable (no repeated words)."""
    name = name.replace("___", " - ")
    name = re.sub(r"_+", " ", name)
    words = name.split()
    if len(words) > 1 and words[0] == words[1]:
        words.pop(0)
    return " ".join(words)


model, class_names = load_model()

tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use camera"])

with tab_upload:
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

with tab_camera:
    st.caption("Hold a single leaf in front of the camera on a plain background.")
    captured = st.camera_input("Take a photo of the leaf")

# camera photo takes priority if both are present
file = captured if captured is not None else uploaded

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    img = np.array(image.resize(IMG_SIZE), dtype=np.float32)
    img = preprocess_input(np.expand_dims(img, axis=0))

    probs = model.predict(img, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]

    best = clean_name(class_names[top3[0]])
    st.success(f"Prediction: **{best}** ({probs[top3[0]] * 100:.2f}% confidence)")

    if probs[top3[0]] < 0.6:
        st.warning(
            "Low confidence. Try a clearer photo: single leaf, plain background, good light. "
            "This model only knows Tomato, Potato and Pepper bell leaves."
        )

    st.subheader("Top 3 predictions")
    for i in top3:
        st.write(f"{clean_name(class_names[i])} - {probs[i] * 100:.2f}%")
        st.progress(float(probs[i]))
