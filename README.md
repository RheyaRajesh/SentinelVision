<div align="center">

# SentinelVision 🛡️📹  
**ATM Surveillance Alert Automation System**

**Real-time ATM security monitoring** powered by **YOLOv8** object detection + **ensemble ML** (SVM, Decision Tree, Random Forest) for intelligent **true/false alert classification**.

</div>

## 🌟 What is SentinelVision?

An intelligent, ML-driven **ATM surveillance system** that:

- 🔥 Detects **fire** & **smoke** instantly
- ⛑️ Checks for **helmet usage** (suspicious masked activity)
- 🧠 Performs **behavioral analysis** to spot anomalies
- 🚨 Filters **false positives** using ensemble classifiers (SVM + DT + RF)
- 📊 Provides real-time alerts + analytics dashboard

Perfect for enhancing ATM security, reducing manual monitoring, and preventing fraud/fire incidents.

## ✨ Key Features

- 🟢 **YOLOv8** powered real-time object detection (fire, smoke, helmet, person, suspicious objects)
- 🧩 **Ensemble ML** post-processing → drastically reduces false alarms
- 📹 Supports **video files**, **webcam**, **RTSP/IP camera feeds**
- 🎨 Beautiful **Streamlit web app** for upload, live view & analytics
- 📈 Dashboard shows: alert count, confidence scores, timeline, false-positive rate
- 🔔 Instant visual + (optional) notification alerts

## 🛠️ Tech Stack

| Component              | Technology                          | Purpose                              |
|------------------------|-------------------------------------|--------------------------------------|
| Object Detection       | YOLOv8 (Ultralytics)                | Fast & accurate detection            |
| Alert Classification   | Scikit-learn (SVM, DT, RF)          | True/False alert filtering           |
| Web Interface          | Streamlit                           | Interactive demo & dashboard         |
| Data Handling          | OpenCV, NumPy, Pandas               | Video processing & feature extraction|
| Visualization          | Matplotlib / Plotly (inside Streamlit) | Analytics charts                  |

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/SentinelVision.git
cd SentinelVision

# 2. Create & activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Important) Download datasets & place in data/
#    → See Datasets section below

# 5. Preprocess data for YOLO
python preprocess.py

# 6. Train models (YOLO + ML ensemble)
python train_yolo.py
python train_ml.py

# 7. Launch the awesome Streamlit app! 🎉
streamlit run app.py
