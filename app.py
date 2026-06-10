import cv2
import streamlit as st
import os
import numpy as np

# -----------------------------
# BASE PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# LOAD MODELS (FIXED PATH)
# -----------------------------
age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "age_deploy.prototxt"),
    os.path.join(MODEL_DIR, "age_net.caffemodel")
)

gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "gender_deploy.prototxt"),
    os.path.join(MODEL_DIR, "gender_net.caffemodel")
)

# -----------------------------
# LABELS
# -----------------------------
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# -----------------------------
# UI
# -----------------------------
st.title("Age & Gender Detection")

run = st.checkbox("Start Camera")

frame_window = st.image([])

camera = cv2.VideoCapture(0)

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# MAIN LOOP
# -----------------------------
while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.426, 87.769, 114.896),
            swapRB=False
        )

        # Gender
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

        # Age
        age_net.setInput(blob)
        age = AGE_LIST[age_net.forward()[0].argmax()]

        label = f"{gender}, {age}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
