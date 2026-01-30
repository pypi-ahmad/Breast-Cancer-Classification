# Breast Cancer Diagnostic AI 🏥

## Overview
A multi-model machine learning system (SVM, XGBoost, RF) designed to detect malignant tumors with high sensitivity. The app provides a medical-grade Streamlit dashboard for prediction, threshold tuning, and exploratory analysis.

## Key Features
- **Sensitivity Tuner**: Real-time decision-threshold adjustment to minimize False Negatives.
- **Multi-Model Engine**: Comparative analysis across 5 different algorithms.
- **Interactive EDA**: Deep dive into tumor features (Radius, Texture, etc.).

## Tech Stack
- Python 3.13
- Streamlit
- Plotly
- Docker
- Scikit-Learn

## Quick Start
### Local
1. Install dependencies:
   pip install -r requirements.txt
2. Run the app:
   streamlit run app.py

### Docker
1. Build the image:
   docker build -t cancer-app .
2. Run the container:
   docker run -p 8501:8501 cancer-app

## Disclaimer
For educational/research purposes only. Not for clinical use.
