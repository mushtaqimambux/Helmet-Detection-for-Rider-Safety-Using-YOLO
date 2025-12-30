# --- VIDEO MODE (Fixed for Preview) ---
elif source == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=['mp4'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        if st.button('Process Video'):
            cap = cv2.VideoCapture(tfile.name)
            st.info("Video process ho rahi hai... niche frames dekhein.")
            
            # --- YE LINE ZAROORI HAI ---
            # Ek khali jagah banayein jahan frames update honge
            frame_placeholder = st.empty() 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Model detection
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # BGR to RGB conversion (Streamlit RGB mangta hai)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # --- YE LINE PREVIEW DIKHAYEGI ---
                frame_placeholder.image(res_rgb, caption="Live Detection", use_container_width=True)
            
            cap.release()
            st.success("Video processing mukammal!")
