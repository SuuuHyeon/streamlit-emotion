# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import predict_emotion

st.set_page_config(page_title="감정 분류기", layout="centered")

st.title("😊 얼굴 감정 분류기")
st.write("얼굴이 포함된 이미지를 업로드하면 감정을 분석해드립니다.")

uploaded_file = st.file_uploader("얼굴 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    emotion, coords = predict_emotion(image_bgr)

    st.subheader("결과:")
    st.write(f"예측된 감정: **{emotion}**")

    if coords:
        x, y, w, h = coords
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)

    st.image(image_np, caption="업로드된 이미지", use_column_width=True)
