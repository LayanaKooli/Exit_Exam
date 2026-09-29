import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
# Make sure 'mudra_classifier_model.h5' is in the same directory as this script
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('mudra_classifier_model.h5')
    return model

model = load_my_model()

# Get class names from your notebook's `class_names` variable (assuming it's defined globally or you hardcode it)
class_names = ['Trisula', 'Musti', 'Sikhara', 'Simhamukha', 'Pataka'] # Replace with your actual class_names
img_size = 128 # The size your model was trained on

# Function to preprocess the image
def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # Read image in BGR format

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img = cv2.resize(img, (img_size, img_size)) # Resize to the model's expected input size
    img = img.astype('float32') / 255.0 # Normalize pixel values
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img

# Streamlit App Title
st.title("Bharatanatyam Mudra Classifier")

st.write("Upload an image of a mudra to get a classification prediction.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(uploaded_file)
    if processed_image is not None:
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        st.success(f"Prediction: **{predicted_class_name}** with {confidence*100:.2f}% confidence.")
        st.write("--- ")
        st.write("All Class Probabilities:")
        # Display all probabilities
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {predictions[0][i]*100:.2f}%")
    else:
        st.error("Could not process the image. Please try another file.")


