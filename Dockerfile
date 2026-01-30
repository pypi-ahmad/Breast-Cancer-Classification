FROM python:3.11-slim

WORKDIR /app

LABEL org.opencontainers.image.title="OncoGuard"
LABEL org.opencontainers.image.description="Streamlit command center for breast tumor classification (Malignant vs Benign) with XAI (SHAP) and multi-model consensus."
LABEL org.opencontainers.image.licenses="MIT"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py models_bundle.pkl sample_data.csv ./

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
