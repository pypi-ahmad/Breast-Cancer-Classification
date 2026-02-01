# Universal ML Command Center 🌍

> **A "No-Code" Adaptive Machine Learning Platform.**
>
> [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
> [![FLAML](https://img.shields.io/badge/FLAML-Microsoft-blue)](https://microsoft.github.io/FLAML/)
> [![SHAP](https://img.shields.io/badge/XAI-SHAP-orange)](https://shap.readthedocs.io/)

This repository hosts a **Universal Classification Engine** that automatically trains high-performance models using **FLAML (AutoML)** and generates a **Dynamic Dashboard** that adapts its UI to your dataset.

Whether you are predicting **Breast Cancer**, **Customer Churn**, or **Loan Default**, this system adapts without requiring you to rewrite the frontend code.

---

## 🚀 Key Features

### 1. 🤖 Automated Machine Learning (AutoML)
Powered by **Microsoft FLAML**, the engine automatically:
- Selects the best algorithms (XGBoost, LightGBM, Random Forest, etc.).
- Tunes hyperparameters within your specified time budget.
- Generates an **Ensemble** of top-performing models for "Consensus Voting".

### 2. 🦎 Adaptive Dashboard
The `app.py` frontend is **dataset-agnostic**. It reads metadata from the trained bundle to automatically:
- Set the App Title.
- Label the classes (e.g., "Malignant" vs "Benign", or "Churn" vs "Stay").
- Generate input fields for the **Symptom Predictor** based on your feature columns.

### 3. 🩺 Symptom Predictor (What-If Simulator)
A sidebar tool allows doctors or analysts to manually input data and get real-time predictions from the best model, complete with probability scores.

### 4. 🧠 Explainable AI (XAI)
Built-in **SHAP (SHapley Additive exPlanations)** integration provides:
- **Global Importance:** Which features matter most across the dataset?
- **Local Explanation:** Why did the model predict "High Risk" for *this specific patient*?

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- [Optional] Virtual Environment recommended.

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration & Training
Open `train_automl.py` and edit the **CONFIGURATION** block at the top:

```python
# --- CONFIGURATION ---
DATA_SOURCE = "sklearn_breast_cancer"  # or path/to/your.csv
TARGET_COLUMN = "target"               # The column to predict
APP_TITLE = "Breast Cancer AI 🏥"      # Dashboard Title
CLASS_LABELS = {0: "Malignant", 1: "Benign"} # Map 0/1 to text
TIME_BUDGET = 90                       # Training time (seconds)
```

Run the engine:
```bash
python train_automl.py
```
*This will train the models and save a `models_bundle.pkl` artifact.*

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
*The app will launch in your browser, fully adapted to your new dataset.*

---

## 📂 Project Structure

- **`train_automl.py`**: The "Backend". Configurable AutoML engine that trains models and saves the bundle.
- **`app.py`**: The "Frontend". A universal Streamlit dashboard that loads the bundle and adapts the UI.
- **`models_bundle.pkl`**: The "Brain". Contains the trained models, scaler, and metadata (Title, Labels, Features).
- **`requirements.txt`**: Dependency list.

---

## 🛡️ License
MIT License
