import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load OpenCV models (UPDATE PATHS if needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Real-Time Face Detection (Streamlit Cloud)")

st.write("Take a picture using your camera below 👇")

# 📸 Camera input (WORKS ON STREAMLIT CLOUD)
img_file = st.camera_input("Capture Image")

if img_file is not None:
    # Convert image to OpenCV format
    image = Image.open(img_file)
    image = np.array(image)

    # Convert RGB → BGR
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert back to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame, channels="RGB")
    st.success(f"Faces detected: {len(faces)}")
