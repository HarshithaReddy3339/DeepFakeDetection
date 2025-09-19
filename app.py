import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once and cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fake_detection_cnn.h5")

model = load_model()

def preprocess_img(img):
    img = img.resize((128,128))  # match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

st.title("🎭 Deepfake Detection App")
st.write("Upload a face image, and the model will classify it as **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_img(img)
    pred = model.predict(img_array)[0][0]

    label = "Fake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1-pred

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence*100:.2f}%")
