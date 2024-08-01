import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os

@st.cache(allow_output_mutation=True)
def load_model_from_drive():
    url = 'https://drive.google.com/file/d/1hygh2cZMHbizNrM895nUxC0pD82InooC/view?usp=drive_link'
    output = 'disease_detection_model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return load_model(output)

model = load_model_from_drive()

st.title("AI-Powered Early Disease Detection Using Medical Imaging")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = int(np.round(prediction[0][0]))

    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {"Positive" if result == 1 else "Negative"}')
