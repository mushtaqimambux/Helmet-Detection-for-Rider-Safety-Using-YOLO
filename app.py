import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# 1. Model Load karein
model = YOLO('best.pt') 

st.set_page_config(page_title="Helmet Detection AI", page_icon="⛑️")
st.title("⛑️ Helmet Detection System")

# Sidebar for confidence
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
    
    # --- Yahan se fix kiya gaya hai ---
    elif source == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        if st.button('Process Video'):
            cap = cv2.VideoCapture(tfile.name)
            st.info("Video process ho rahi hai... niche frames dekhein.")
            
            # Ye placeholder preview dikhane ke liye zaroori hai
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Model Prediction
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # BGR to RGB conversion for Streamlit
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Live Preview Update
                frame_placeholder.image(res_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
            st.success("Video processing mukammal!")
