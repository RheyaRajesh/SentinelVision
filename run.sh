#!/bin/bash
# Make executable: chmod +x run.sh
echo "Starting ATM Surveillance App..."
streamlit run app.py --server.port=8501 --server.headless=true