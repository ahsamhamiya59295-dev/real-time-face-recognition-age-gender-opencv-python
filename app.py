import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FaceInsight AI",
    layout="centered",
    page_icon="🤖"
)

# ---------------- NEON UI CSS ----------------
st.markdown("""
<style>

body {
    background-color: #0a0f1c;
}

.stApp {
    background: linear-gradient(135deg, #050a18, #0a0f1c);
}

/* Title */
h1 {
    text-align: center;
    color: #00e5ff;
    font-size: 42px;
    font-weight: 900;
    text-shadow: 0 0 15px #00e5ff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #00ff88;
    font-size: 16px;
    margin-bottom: 25px;
    text-shadow: 0 0 10px #00ff88;
}

/* Box */
.box {
    background: rgba(10, 20, 40, 0.7);
    border: 1px solid #00e5ff;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 20px #00e5ff33;
}

/* Results */
.result {
    color: #00ff88;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    text-shadow: 0 0 10px #00ff88;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00e5ff, #00ff88);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    box-shadow: 0 0 15px #00e5ff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🤖 FaceInsight AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Face • Age • Gender Detection </p>", unsafe_allow_html=True)

# ---------------- MODEL PATH ----------------
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

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# ---------------- UI BOX ----------------
st.markdown("<div class='box'>", unsafe_allow_html=True)

img_file = st.camera_input("Logging Face Image")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if img_file is not None:

    image = Image.open(img_file)
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    age_out = "-"
    gender_out = "-"

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4, 87.7, 114.9),
                swapRB=False
            )

            gender_net.setInput(blob)
            gender_out = GENDER_LIST[gender_net.forward().argmax()]

            age_net.setInput(blob)
            age_out = AGE_LIST[age_net.forward().argmax()]

            label = f"{gender_out}, {age_out}"

        except:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

    st.image(frame, channels="RGB")

    st.markdown(f"<p class='result'>👤 Faces: {len(faces)} | {gender_out} | {age_out}</p>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<p style='text-align:center;color:#00e5ff;'>⚡ Powered by AHSAM KARIM </p>", unsafe_allow_html=True)
