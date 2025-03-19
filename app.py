import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from huggingface_hub import hf_hub_download


# Model Paths
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "LSTM_model.keras")


# Download model from Hugging Face
#CNN_MODEL_PATH='....'


print("Model loaded successfully!")


# Define class labels
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect diseases.")

# Model Selection Dropdown (No Default)
# Model Selection Dropdown (No Default)
st.write("🔍 **Select a Model:**")
model_choice = st.selectbox("", ["Select Model", "LSTM", "CNN"], index=0)

if model_choice == "CNN":
    st.warning("⚠️ Please download the CNN model from drive and place it in the correct folder.")
elif model_choice != "Select Model":
    # Load the selected model
    model_path = LSTM_MODEL_PATH if model_choice == "LSTM" else CNN_MODEL_PATH
    model = tf.keras.models.load_model(model_path)



    # Upload Image
    uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Adjust Image Resizing Based on the Model's Expected Input Shape
        if model_choice == "LSTM":
            target_size = (128,128)  # Adjusted to match 41472 input
        else:  # CNN Model
            target_size = (224,224)

        image_resized = image.resize(target_size)  
        input_arr = np.array(image_resized) / 255.0  # Normalize pixel values

        # Adjust input shape based on the model
        if model_choice == "LSTM":
            input_arr = input_arr.reshape(1, 128, 128, 3)
        else:
            input_arr = np.expand_dims(input_arr, axis=0)  # (1, 224, 224, 3)
            feature_extractor = tf.keras.Model(model.input, model.layers[-2].output)  # Get feature layer output
            input_arr = feature_extractor.predict(input_arr) 
           


        

        # Debugging Statements
        print("Model expected input shape:", model.input_shape)
        print("Actual input shape:", input_arr.shape)

        # Make Prediction
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)
        disease_name = class_names[result_index]

        # Create Two Columns
        col1, col2 = st.columns(2)
        st.write("") 
        with col1:
            st.header("📥 Uploaded Image")
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.header("🔍 Prediction")
            st.image(image, caption=f"Predicted: {disease_name}", use_container_width=True)
            st.success(f"🌿 **Disease Detected:** {disease_name}")
else:
    st.warning("⚠️ Please select a model to proceed.")
