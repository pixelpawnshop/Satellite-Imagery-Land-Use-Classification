"""
Streamlit app for EuroSAT Land Use Classification
Frontend for uploading RGB satellite images and displaying model predictions.
Connects to FastAPI backend at /predict.
"""
import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"  # Change to your server's public URL when deployed

st.title("EuroSAT Land Use Classification Demo")
st.write("Upload a Sentinel-2 RGB image (JPG/PNG, 64x64) to get a land use prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    if st.button("Predict Land Use Class"):
        with st.spinner("Predicting..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['predicted_class']}")
                st.info(f"Confidence: {result['confidence']}")
            else:
                st.error(f"Error: {response.text}")
