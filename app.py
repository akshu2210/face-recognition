import cv2
import os
import numpy as np
import streamlit as st

st.title("Missing Person Tracking System (Accurate AI Version)")

# Load Haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Prepare training data
path = 'dataset'
faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(path):
    person_path = os.path.join(path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🔥 Improved detection
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 🔥 Fallback if no face detected
        if len(detected_faces) == 0:
            face = cv2.resize(gray, (200, 200))
            faces.append(face)
            labels.append(current_label)
        else:
            for (x, y, w, h) in detected_faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                faces.append(face)
                labels.append(current_label)

    current_label += 1

# Debug check
st.write("Faces loaded:", len(faces))

# Train model
if len(faces) == 0:
    st.error("❌ No training data found. Check dataset folder.")
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    st.write("Upload Image")

    uploaded_file = st.file_uploader("Choose image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🔥 Improved detection
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(detected_faces) == 0:
            # 🔥 Fallback prediction
            face = cv2.resize(gray, (200, 200))
            label, confidence = recognizer.predict(face)

            if confidence < 80:
                name = label_map[label]
                st.success(f"✅ Match Found: {name} (Confidence: {confidence:.2f})")
            else:
                st.error("❌ No Match Found")

        else:
            for (x, y, w, h) in detected_faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                label, confidence = recognizer.predict(face)

                # 🔥 Improved threshold
                if confidence < 80:
                    name = label_map[label]
                    st.success(f"✅ Match Found: {name} (Confidence: {confidence:.2f})")
                else:
                    st.error("❌ No Match Found")