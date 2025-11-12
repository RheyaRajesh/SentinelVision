import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inference import process_video, process_live_feed
import os
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="ATM Surveillance Alert System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS (Black + Radiant Blue) ======================
st.markdown("""
    <style>
    .main {background-color: #000000;}
    .stApp {background-color: #000000;}
    .stMarkdown {color: #00BFFF;}
    .stTextInput > div > div > input {background-color: #111111; color: #00BFFF;}
    .stButton > button {background-color: #00BFFF; color: #000000; font-weight: bold;}
    .stFileUploader {background-color: #111111; border: 2px dashed #00BFFF;}
    .stFileUploader > div > div {color: #00BFFF;}
    .stDataFrame {background-color: #111111; color: #00BFFF;}
    .stMetric {color: #00BFFF;}
    </style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.title("ATM Surveillance Alert Automation System")
st.markdown("*Real-time detection and validation of true alerts from video feeds.*")

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
upload_mode = st.sidebar.radio("Input Mode", ["Upload Video", "Live Camera"])

# ====================== MAIN: UPLOAD VIDEO ======================
if upload_mode == "Upload Video":
    st.subheader("Upload Video File")
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov'],
        help="Max 200MB"
    )

    if uploaded_file is not None:
        # Save uploaded file to temp
        with st.spinner("Saving video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name
            tfile.close()

        # Process video
        with st.spinner("Processing video with AI detection..."):
            try:
                alerts_df, stats = process_video(tfile_path, max_frames=100)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                alerts_df = pd.DataFrame()
                stats = {'true_count': 0, 'false_count': 0}

        # ====================== DISPLAY RESULTS ======================
        col1, col2 = st.columns([1, 1])

        with col1:
            # Show last processed frame
            last_frame_path = 'outputs/processed_videos/last_frame.jpg'
            if os.path.exists(last_frame_path):
                st.image(last_frame_path, caption="Latest Frame with Detections", use_column_width=True)
            else:
                st.info("No frame to display.")

        with col2:
            # Show processed video
            output_video_path = 'outputs/processed_videos/output.mp4'
            if os.path.exists(output_video_path):
                st.video(output_video_path)
            else:
                st.info("No processed video available.")

        # ====================== ALERTS TABLE ======================
        if not alerts_df.empty:
            st.subheader("Detected Alerts")
            st.dataframe(alerts_df.style.highlight_max(axis=0), use_container_width=True)

            # ====================== ANALYTICS CHART ======================
            st.subheader("Alert Analytics")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
            counts = alerts_df['type'].value_counts()
            bars = ax.bar(counts.index, counts.values, color='#00BFFF', edgecolor='white', linewidth=1.2)
            ax.set_title("Alert Type Distribution", color='white', fontsize=16, pad=20)
            ax.set_xlabel("Alert Type", color='white', fontsize=12)
            ax.set_ylabel("Count", color='white', fontsize=12)
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')
            ax.grid(True, axis='y', color='gray', linestyle='--', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)

            # ====================== METRICS ======================
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts", len(alerts_df), delta=None)
            with col2:
                st.metric("True Alerts", stats['true_count'], delta=None)
            with col3:
                st.metric("False Alerts", stats['false_count'], delta=None)

            # ====================== SAVE LOGS ======================
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = "outputs/logs"
            os.makedirs(log_dir, exist_ok=True)  # Create logs folder
            log_path = os.path.join(log_dir, f"alerts_{timestamp}.csv")
            alerts_df.to_csv(log_path, index=False)
            st.success(f"Alerts logged: `{log_path}`")

            # Download button
            csv = alerts_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Alerts CSV",
                data=csv,
                file_name=f"alerts_{timestamp}.csv",
                mime="text/csv"
            )

        else:
            st.info("No alerts detected in this video.")
            st.warning("Try a video with fire, smoke, or people for better results.")

        # Clean up temp file
        try:
            os.unlink(tfile_path)
        except:
            pass

# ====================== MAIN: LIVE CAMERA ======================
elif upload_mode == "Live Camera":
    st.subheader("Live Camera Feed")
    if st.button("Start Live Surveillance"):
        st.warning("Webcam access in Streamlit requires HTTPS or localhost. Simulating with sample video.")
        
        # Use a sample image or video for demo
        sample_img = "data/fire/fire1.jpg"  # Change to your actual image
        if os.path.exists(sample_img):
            img = cv2.imread(sample_img)
            alerts_df, stats = process_video(sample_img, max_frames=1)  # Treat image as 1-frame video
            if not alerts_df.empty:
                st.success("Live simulation: Alert detected!")
                st.dataframe(alerts_df)
            else:
                st.info("No alert in simulation.")
        else:
            st.info("No sample image found. Showing mock data.")
            alerts_df = pd.DataFrame({
                'timestamp': ['10:30:15'], 'type': ['fire'], 'confidence': [0.92], 'is_true': [True]
            })
            st.dataframe(alerts_df)

# ====================== FOOTER ======================
st.sidebar.markdown("---")
st.sidebar.info(
    "**Features:**\n"
    "- Fire, Smoke, Helmet, Mask Detection\n"
    "- True/False Alert Validation\n"
    "- Real-time Analytics\n"
    "- CSV Export & Logging"
)
st.sidebar.markdown("Made with **YOLOv8 + Ensemble ML**")