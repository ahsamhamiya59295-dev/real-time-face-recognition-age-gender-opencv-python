import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face AI Pro",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS (STUNNING UI) ----------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    h1 {
        text-align: center;
        color: #00ffe1;
        font-size: 42px;
        font-weight: 800;
    }

    .subtitle {
        text-align: center;
        color: #9aa4b2;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .box {
        background: linear-gradient(145deg, #111827, #0b1220);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #1f2937;
        box-shadow: 0 0 20px rgba(0,255,225,0.15);
    }

    .result {
        font-size: 20px;
        font-weight: bold;
        color: #00ff9d;
        text-align: center;
        margin-top: 10px;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🤖 Face AI Pro System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI detects Face • Age • Gender in real time</p>", unsafe_allow_html=True)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------- LOAD MODELS ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "age_deploy.prototxt"),
    os.path.join(MODEL_DIR, "age_net.caffemodel")
)

gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "gender_deploy.prototxt"),
    os.path.join(MODEL_DIR, "gender_net.caffemodel")
)

# ---------------- LABELS ----------------
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# ---------------- UI CARD ----------------
st.markdown("<div class='box'>", unsafe_allow_html=True)

img_file = st.camera_input("📸 Capture Image")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESSING ----------------
if img_file is not None:

    image = Image.open(img_file)
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4, 87.7, 114.9),
                swapRB=False
            )

            gender_net.setInput(blob)
            gender = GENDER_LIST[gender_net.forward().argmax()]

            age_net.setInput(blob)
            age = AGE_LIST[age_net.forward().argmax()]

            label = f"{gender}, {age}"

        except:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    st.image(frame, use_container_width=True)

    st.markdown(
        f"<p class='result'>Faces detected: {len(faces)}</p>",
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown("<p class='footer'>Powered by AI • Streamlit • OpenCV</p>", unsafe_allow_html=True)
