import os
import cv2
import numpy as np
import onnxruntime as ort

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ONNX 모델 로드
session = ort.InferenceSession("model/emotion_model.onnx")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cascade 분류기 경로 확인
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(haar_path):
        raise FileNotFoundError(f"Haar cascade 파일을 찾을 수 없습니다: {haar_path}")

    face_cascade = cv2.CascadeClassifier(haar_path)

    # 얼굴 검출 민감도 조정
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    # 얼굴 검출 실패 시 fallback (이미지 중앙)
    if len(faces) == 0:
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        # 이미지가 너무 작으면 crop 실패할 수 있으므로 체크
        if h < 48 or w < 48:
            return None, None
        roi_gray = gray[cy - 24:cy + 24, cx - 24:cx + 24]
        coords = None
    else:
        x, y, w, h = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        coords = (x, y, w, h)

    # 정규화 및 차원 조정
    roi_gray = roi_gray.astype("float32") / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=(0, -1))  # shape: (1, 48, 48, 1)

    return roi_gray, coords

def predict_emotion(image):
    processed_img, coords = preprocess_image(image)
    if processed_img is None:
        return "얼굴을 감지하지 못했습니다.", None

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: processed_img})
    prediction = outputs[0]

    emotion = emotion_labels[np.argmax(prediction)]
    return emotion, coords
