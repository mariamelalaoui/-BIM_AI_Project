import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Streamlit UI
st.title("🚧 BIM & AI-Driven Risk Assessment for Construction Projects")
st.write("Detect safety violations in construction sites using AI.")

# Choose input method
input_option = st.radio("Choose input method:", ("Upload Image", "Upload Video", "Live Webcam"))

# 🖼️ **Image Upload & Processing**
if input_option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Run YOLO model on image
        results = model(image)

        # Draw bounding boxes + labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(box.conf[0].item(), 2)  # Confidence score
                label = model.names[int(box.cls[0])]  # Class name (e.g., "Helmet", "Person")
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put label text
                cv2.putText(image, f"{label} {confidence}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert color format for Streamlit
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Processed Image with Labels", use_column_width=True)

# 🎥 **Video Upload & Processing**
elif input_option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video:
        # Save video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        # Display uploaded video
        st.video(temp_video.name)

        # Run YOLO on video
        with st.spinner("Processing video..."):
            results = model(temp_video.name, save=True)
        
        st.success("✅ Processing complete! Check the results.")

# 📷 **Live Webcam (Not Implemented Yet)**
elif input_option == "Live Webcam":
    st.write("⚠️ Live webcam detection is under development.")
