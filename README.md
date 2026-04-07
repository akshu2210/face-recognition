# face-recognition
AI-based missing person tracking system using OpenCV and LBPH
# 🔍 Missing Person Tracking System (AI)

This project is an AI-based system to detect and recognize missing persons using face recognition.

## 🚀 Features
- Face detection using Haarcascade
- Face recognition using LBPH algorithm
- Upload image and detect person
- Works with multiple images per person

## 🛠 Technologies Used
- Python
- OpenCV
- Streamlit
- NumPy

## 📂 Project Structure
missing_person_ai/
│── app.py
│── dataset/
│── haarcascade_frontalface_default.xml

## ▶️ How to Run

1. Clone repository:
git clone https://github.com/akshu2210/face-recognition.git

2. Install dependencies:
pip install opencv-contrib-python streamlit numpy

3. Run the project:
streamlit run app.py

## 📸 Output
- Detects face from uploaded image
- Matches with dataset
- Displays person name with confidence

## ⚠️ Limitations
- Requires clear face images
- Sensitive to lighting conditions

## 🔮 Future Improvements
- Deep learning-based recognition (FaceNet)
- Live webcam tracking
- Cloud deployment

---
✨ Developed as part of IPD Project
