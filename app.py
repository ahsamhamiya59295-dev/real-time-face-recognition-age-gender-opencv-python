import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Face Detection System",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }

    h1 {
        text-align: center;
        color: #00ffe1;
        font-size: 40px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #aaa;
        font-size: 18px;
        margin-bottom: 20px;
    }

    .box {
        border: 2px solid #00ffe1;
        padding: 15px;
        border-radius: 15px;
        background-color: #111827;
        box-shadow: 0px 0px 15px #00ffe1;
    }

    .result {
        font-size: 20px;
        color: #00ff9d;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🤖 AI Face Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture an image and detect faces instantly</p>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- INPUT ----------------
st.markdown("<div class='box'>", unsafe_allow_html=True)
img_file = st.camera_input("📸 Capture Image")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESSING ----------------
if img_file is not None:

    image = Image.open(img_file)
    image = np.array(image)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame, use_container_width=True)

    st.markdown(
        f"<p class='result'>✅ Faces Detected: {len(faces)}</p>",
        unsafe_allow_html=True
    )
