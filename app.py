import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt

# ---------------- MODEL PATH ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

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


# ---------------- UPLOAD WIDGET ----------------
class UploadWidget(QtWidgets.QFrame):
    image_dropped = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedSize(360, 360)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        self.icon = QtWidgets.QLabel("☁️")
        self.icon.setAlignment(QtCore.Qt.AlignCenter)
        self.icon.setFixedHeight(120)
        layout.addWidget(self.icon)

        self.title = QtWidgets.QLabel("Click or Drag & Drop Image")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.title)

        self.subtitle = QtWidgets.QLabel("JPG, PNG, WEBP supported")
        self.subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.subtitle)

        self.analyze_btn = QtWidgets.QPushButton("Analyze Face")
        layout.addWidget(self.analyze_btn, alignment=QtCore.Qt.AlignCenter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self.image_dropped.emit(path)


# ---------------- IMAGE VIEW ----------------
class ResultImageWidget(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(320, 240)

    def set_image(self, pil_image):
        w, h = self.width(), self.height()
        pil_image = pil_image.copy()
        pil_image.thumbnail((w, h))
        pix = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_image))
        self.setPixmap(pix)


# ---------------- MAIN APP ----------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FaceInsight AI")
        self.setFixedSize(820, 460)

        layout = QtWidgets.QHBoxLayout(self)

        # LEFT UI (YOUR DESIGN)
        self.upload = UploadWidget()
        layout.addWidget(self.upload)

        # RIGHT PANEL
        right = QtWidgets.QVBoxLayout()

        self.result_image = ResultImageWidget()

        self.age_label = QtWidgets.QLabel("Age: -")
        self.gender_label = QtWidgets.QLabel("Gender: -")
        self.face_label = QtWidgets.QLabel("Faces: -")

        right.addWidget(self.result_image)
        right.addWidget(self.age_label)
        right.addWidget(self.gender_label)
        right.addWidget(self.face_label)

        layout.addLayout(right)

        # DATA
        self.image = None

        # SIGNALS
        self.upload.image_dropped.connect(self.load_image)
        self.upload.analyze_btn.clicked.connect(self.analyze)

    # ---------------- LOAD IMAGE ----------------
    def load_image(self, path):
        self.image = Image.open(path).convert("RGB")
        self.result_image.set_image(self.image)

    # ---------------- AI ANALYSIS ----------------
    def analyze(self):

        if self.image is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Please upload image first")
            return

        frame = np.array(self.image)
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

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        # UPDATE UI
        self.result_image.set_image(Image.fromarray(frame))
        self.age_label.setText(f"Age: {age_out}")
        self.gender_label.setText(f"Gender: {gender_out}")
        self.face_label.setText(f"Faces: {len(faces)}")


# ---------------- RUN ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
