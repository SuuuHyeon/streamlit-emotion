import streamlit as st
import numpy as np
from PIL import Image
from utils.predictor import predict_emotion

# Streamlit Cloud 감지 여부
IS_CLOUD = "streamlit_app" in __file__

st.title("😊 실시간 감정 분류기")

# 로컬 환경일 경우 웹캠 허용
if not IS_CLOUD:
    from utils.camera import get_frame
    if st.button("웹캠으로 사진 찍기"):
        frame = get_frame()
        if frame is not None:
            st.image(frame, caption="캡처된 이미지", use_column_width=True)
            emotion, confidence = predict_emotion(frame)
            st.success(f"예측 감정: {emotion} ({confidence:.2f})")
        else:
            st.error("웹캠을 사용할 수 없습니다.")
else:
    st.info("현재는 Streamlit Cloud 환경입니다. 이미지를 업로드해주세요.")

# 공통 이미지 업로드 처리
uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    image_array = np.array(image)
    emotion, confidence = predict_emotion(image_array)
    st.success(f"예측 감정: {emotion} ({confidence:.2f})")
