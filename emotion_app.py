
import streamlit as st
import numpy as np
from PIL import Image
from utils.predictor import predict_emotion

# Streamlit Cloud 여부 판단 (파일명 기준)
IS_CLOUD = "streamlit_app" in __file__

st.title("😊 실시간 감정 분류기")

# 로컬 웹캠 테스트 기능 (Streamlit Cloud에서는 작동 안함)
if not IS_CLOUD:
    from utils.camera import get_frame
    if st.button("웹캠으로 사진 찍기"):
        frame = get_frame()
        if frame is not None:
            st.image(frame, caption="캡처된 이미지", use_column_width=True)
            emotion, confidence = predict_emotion(frame)
            st.success(f"예측 감정: {emotion} ({confidence:.2f})")
        else:
            st.error("웹캠에서 영상을 가져올 수 없습니다.")
else:
    st.info("Streamlit Cloud에서는 웹캠이 지원되지 않습니다. 이미지를 업로드해주세요.")

# 이미지 업로드 기능 (클라우드 & 로컬 공통)
uploaded = st.file_uploader("이미지를 업로드해보세요", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    image_array = np.array(image)
    emotion, confidence = predict_emotion(image_array)
    st.success(f"예측 감정: {emotion} ({confidence:.2f})")
