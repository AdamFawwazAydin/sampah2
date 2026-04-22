import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Klasifikasi Sampah")

# ======================
# DOWNLOAD MODEL
# ======================
MODEL_PATH = "model_sampah.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    
    file_id = "1fs5cqFvZyXorbs6fZaxWvGOrKtFfodlb"
    url = f"https://drive.google.com/file/d/1fs5cqFvZyXorbs6fZaxWvGOrKtFfodlb/view?usp=sharing"
    
    gdown.download(url, MODEL_PATH, quiet=False)

# ======================
# LOAD MODEL (SAFE)
# ======================
@st.cache_resource
def load_ml_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

model = load_ml_model()

# ======================
# PREPROCESS
# ======================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================
# UI
# ======================
st.title("♻️ Klasifikasi Sampah CNN")
st.write("Organik vs Anorganik")

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Upload")

    img = preprocess_image(image)
    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.success("Anorganik")
    else:
        st.success("Organik")

# ======================
# CAMERA
# ======================
st.write("### 📷 Kamera")

camera_image = st.camera_input("Ambil Foto")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Hasil Kamera")

    img = preprocess_image(image)
    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.success("Anorganik")
    else:
        st.success("Organik")