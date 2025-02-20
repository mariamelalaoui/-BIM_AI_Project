import streamlit as st
import tempfile
import os
import cv2
import re
import numpy as np
from ultralytics import YOLO

def get_generated_video(run_folder='runs/detect'):
    # List all directories in the run folder
    directories = [d for d in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, d))]
    
    # Filter directories that match the pattern 'predict + number'
    predict_dirs = [d for d in directories if re.match(r'^predict\d+$', d)]
    
    # Sort the filtered directories by creation time
    predict_dirs.sort(key=lambda d: os.path.getctime(os.path.join(run_folder, d)), reverse=True)
    # Get the path of the most recent predict folder
    latest_predict_folder = os.path.join(run_folder, predict_dirs[0])

    # Get the path of the video file in the predict folder
    mp4_files = [f for f in os.listdir(latest_predict_folder) if re.match(r'.*\.mp4$', f)]
    if mp4_files:
        video_file = os.path.join(latest_predict_folder, mp4_files[0])
    else:
        video_file = None

    # Return the video file path
    return video_file
    


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
        st.image(image, caption="Processed Image with Labels", use_container_width=True)

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
        video = get_generated_video()
        st.video(video)

# 📷 **Live Webcam (Not Implemented Yet)**
elif input_option == "Live Webcam":
    st.write("📷 **Live webcam detection is running...**")

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Streamlit live video display
    stframe = st.empty()

    stopButton = st.button("Stop Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection on the frame
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                label = result.names[int(box.cls[0])]  # Get class name
                confidence = box.conf[0].item()  # Confidence score

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display processed frame in Streamlit
        stframe.image(frame, channels="RGB", use_container_width=True)

        # Stop when user clicks "Stop" button
        if stopButton:
            cap.release()
            cv2.destroyAllWindows()

    
