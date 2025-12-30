import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# 1. Model Load karein
# Ensure karein ke aapki helmet model file ka naam 'best.pt' hai aur isi folder mein hai
model = YOLO('best.pt') 

st.set_page_config(page_title="Helmet Detection AI", page_icon="⛑️")
st.title("⛑️ Helmet Detection System")

# Sidebar for confidence setting
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.40)

# Input selection
source = st.radio("Select Media Type", ("Image", "Video"))
uploaded_file = st.file_uploader(f"Upload {source}", type=['jpg','png','jpeg','mp4'])

if uploaded_file is not None:
    if source == "Image":
        # Image Processing
        file_bytes = uploaded_file.read()
        results = model.predict(source=file_bytes, conf=conf_threshold)
        st.image(results[0].plot(), caption="Detected Results", use_container_width=True)
    
    else:
        # Video Processing
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st.info("Video process ho rahi hai... niche frames dekhein.")
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Helmet Detection on each frame
            res = model.predict(frame, conf=conf_threshold, verbose=False)
            # Display frame with boxes
            frame_placeholder.image(res[0].plot(), channels="BGR")
            
        cap.release()
        st.success("Video processing mukammal!")