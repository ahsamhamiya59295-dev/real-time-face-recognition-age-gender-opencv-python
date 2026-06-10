import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# ---------------- UI ----------------
st.set_page_config(page_title="AI Face Age Gender", layout="centered")
st.title("🤖 AI Face + Age + Gender Detection")

st.write("Capture an image and the AI will detect faces, age, and gender.")

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

# ---------------- CAMERA INPUT ----------------
img_file = st.camera_input("📸 Capture Image")

if img_file is not None:

    # Convert image
    image = Image.open(img_file)
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4, 87.7, 114.9),
                swapRB=False
            )

            # Gender prediction
            gender_net.setInput(blob)
            gender = GENDER_LIST[gender_net.forward().argmax()]

            # Age prediction
            age_net.setInput(blob)
            age = AGE_LIST[age_net.forward().argmax()]

            label = f"{gender}, {age}"

        except:
            label = "Unknown"

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    st.image(frame, channels="RGB")
    st.success(f"Faces detected: {len(faces)}")
