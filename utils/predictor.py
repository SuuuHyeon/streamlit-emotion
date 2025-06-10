
import numpy as np
import tensorflow as tf

# 모델 불러오기
model = tf.keras.models.load_model("emotion_model/emotion_model.h5")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(frame):
    resized = tf.image.resize(frame, [48, 48])
    gray = tf.image.rgb_to_grayscale(resized)
    normalized = gray / 255.0
    input_tensor = tf.expand_dims(normalized, axis=0)
    predictions = model.predict(input_tensor)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence
