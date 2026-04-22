import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# ======================
# DOWNLOAD MODEL
# ======================
MODEL_PATH = "model_sampah.h5"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    url = "https://drive.google.com/file/d/1fs5cqFvZyXorbs6fZaxWvGOrKtFfodlb/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

# ======================
# LOAD MODEL
# ======================
def predict_dummy(img):
    return "Organik"  # sementara untuk UI

# ======================
# PREPROCESS
# ======================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================
# UI
# ======================
st.title("♻️ Klasifikasi Sampah (CNN)")
st.write("Organik vs Anorganik")

# ======================
# UPLOAD GAMBAR
# ======================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Upload", use_column_width=True)

    img = preprocess_image(image)
    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.success("Anorganik")
    else:
        st.success("Organik")

# ======================
# WEBCAM
# ======================
st.write("### 📷 Ambil dari Kamera")

camera_image = st.camera_input("Ambil Foto")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Hasil Kamera", use_column_width=True)

    img = preprocess_image(image)
    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.success("Anorganik")
    else:
        st.success("Organik")