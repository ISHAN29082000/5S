import streamlit as st
import numpy as np
from PIL import Image

# Check if the ultralytics package is installed
try:
    from ultralytics import YOLO
    st.write("Ultralytics library is installed!")
except ImportError:
    st.write("Ultralytics library is NOT installed!")

# Display title and instructions
st.title("Coke and Cryolite Scanner")
st.write("""
    This app uses YOLO to detect coke and cryolite in a scanned image and classify the proportion of coke.
    Please upload an image to get started.
""")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Run YOLO object detection on the uploaded image
    model = YOLO('yolov8n.pt')  # Replace with the correct model path

    # Inference
    results = model(img)

    # Check results and display them
    if results:
        # Annotate and display results
        annotated_img = results[0].plot()  # Assuming `results[0]` has the annotated image
        st.image(annotated_img, caption="Annotated Image", use_container_width=True)

        # Display detected classes and their confidence
        for det in results[0].boxes.data:
            class_id, confidence = det[5].item(), det[4].item()
            st.write(f"Class ID: {class_id}, Confidence: {confidence:.2f}")

    else:
        st.write("No objects detected in the image.")

else:
    st.write("Please upload an image to analyze.")
