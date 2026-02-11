import cv2
import numpy as np
import os

# -----------------------------
# File Checks
# -----------------------------
required_files = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "models/face_recognizer.yml",
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel",
    "models/gender_deploy.prototxt",
    "models/gender_net.caffemodel"
]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Required file not found: {f}")

# -----------------------------
# Load Haar Cascade
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise Exception("Failed to load Haar Cascade XML file.")

# -----------------------------
# Load LBPH Recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_recognizer.yml")

# -----------------------------
# Age and Gender Detection
# -----------------------------
AGE_BUCKETS = ['(0-2)','(4-6)','(8-12)','(15-20)',
               '(25-32)','(38-43)','(48-53)','(60-100)']
GENDER_LIST = ['Male', 'Female']

age_net = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    "models/gender_deploy.prototxt",
    "models/gender_net.caffemodel"
)

# -----------------------------
# Load Dataset Labels
# -----------------------------
labels = {}
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

for idx, name in enumerate(os.listdir(dataset_path)):
    labels[idx] = name

# -----------------------------
# Start Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open webcam")

# -----------------------------
# Real-time Face Recognition Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        # Predict Name
        label, confidence = recognizer.predict(face_gray)
        name = labels[label] if confidence < 70 else "Unknown"

        # Prepare blob for Age/Gender
        blob = cv2.dnn.blobFromImage(
            face_color, 1.0, (227, 227),
            (78.426, 87.768, 114.895), swapRB=False
        )

        # Gender Prediction
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

        # Age Prediction
        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward()[0].argmax()]

        # Display on Frame
        text = f"{name} | {gender} | {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
