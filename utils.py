# utils.py
import cv2
import numpy as np
import onnxruntime as ort

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ONNX 모델 로드
session = ort.InferenceSession("model/emotion_model.onnx")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype("float32") / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=(0, -1))  # (1, 48, 48, 1)

    return roi_gray, (x, y, w, h)

def predict_emotion(image):
    processed_img, coords = preprocess_image(image)
    if processed_img is None:
        return "얼굴을 감지하지 못했습니다.", None

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: processed_img})
    prediction = outputs[0]

    emotion = emotion_labels[np.argmax(prediction)]
    return emotion, coords
