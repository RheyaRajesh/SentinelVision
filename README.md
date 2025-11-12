# ATM Surveillance Alert Automation System

## Overview
A ML-based system for ATM surveillance using YOLOv8 for detection and ensemble ML (SVM, DT, RF) for true/false alert classification. Features: Fire/smoke detection, helmet check, behavioral analysis.

## Setup
1. Install: `pip install -r requirements.txt`
2. Download datasets (see below).
3. Preprocess: `python preprocess.py`
4. Train: `python train_yolo.py` & `python train_ml.py`
5. Run: `streamlit run app.py`

## Structure
- `data/`: Raw datasets.
- `data_yolo/`: Processed for YOLO.
- `models/`: Trained weights.
- `utils/`: Helpers.

## Datasets
See download section below.

## Demo
Upload video in app for real-time alerts + analytics.

