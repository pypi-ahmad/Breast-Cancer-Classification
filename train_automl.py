"""
Universal Classification Training Engine.

This script uses Microsoft FLAML to automatically train and tune multiple classification models
(LGBM, XGBoost, Random Forest, etc.) on a specified dataset. It saves the trained models,
scaler, and metadata into a pickle bundle for the dashboard app.
"""
import warnings
import joblib
import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- CONFIGURATION ---
# -----------------------------------------------------------------------------
# 1. DATA_SOURCE:
#    - To use the built-in Breast Cancer dataset: "sklearn_breast_cancer"
#    - To use your own data: Provide the path to a CSV file (e.g., "data/heart_disease.csv")
# 2. TARGET_COLUMN:
#    - The exact name of the column you want to predict (e.g., "diagnosis", "churn", "is_fraud")
# 3. APP_TITLE:
#    - The title displayed on the Dashboard (e.g., "Customer Churn Predictor 📉")
# 4. CLASS_LABELS:
#    - Map numeric targets to human-readable labels.
#      Example: {0: "Stayed", 1: "Churned"} (Ensure these match your data encoding)
# -----------------------------------------------------------------------------
DATA_SOURCE = "sklearn_breast_cancer"  # Options: 'sklearn_breast_cancer' or 'path/to/data.csv'
TARGET_COLUMN = "target"              # Only needed if using CSV
APP_TITLE = "Breast Cancer AI 🏥"
CLASS_LABELS = {0: "Malignant", 1: "Benign"} # Map numeric targets to text
TEST_SIZE = 0.2
TIME_BUDGET = 90 # Seconds per model

def load_data() -> pd.DataFrame:
    """
    Loads data from the configured source (Sklearn or CSV).

    Returns:
        pd.DataFrame: The loaded dataset containing features and the target column.

    Raises:
        FileNotFoundError: If the DATA_SOURCE path is invalid.
    """
    if DATA_SOURCE == "sklearn_breast_cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        print("✅ Loaded sklearn breast cancer dataset")
        return df
    else:
        try:
            df = pd.read_csv(DATA_SOURCE)
            print(f"✅ Loaded data from {DATA_SOURCE}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Could not find file: {DATA_SOURCE}")

def train_flaml_model(x_train: np.ndarray, y_train: pd.Series, estimator_name: str, time_budget: int) -> AutoML:
    """
    Trains a single FLAML AutoML model restricted to a specific estimator type.

    Args:
        x_train (np.ndarray): Scaled training features.
        y_train (pd.Series): Training labels.
        estimator_name (str): The specific FLAML estimator to use (e.g., 'lgbm', 'rf').
        time_budget (int): Time in seconds allocated for tuning.

    Returns:
        AutoML: The trained AutoML object.
    """
    automl = AutoML()
    
    settings = {
        "time_budget": time_budget,  # total running time in seconds
        "metric": 'roc_auc',         # optimization metric
        "task": 'classification',    # task type
        "estimator_list": [estimator_name], # Restrict to specific algorithm
        "log_file_name": f"flaml_{estimator_name}.log",
        "verbose": 0,
        "seed": 42,
    }
    
    print(f"   ⏳ Tuning {estimator_name}...")
    automl.fit(X_train=x_train, y_train=y_train, **settings)
    print(f"      - Best config: {automl.best_config}")
    print(f"      - Best accuracy: {1 - automl.best_loss:.4f}")
    return automl

def evaluate_models(models: dict, x_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
    """
    Calculates performance metrics for all trained models.

    Args:
        models (dict): Dictionary of {name: model_object}.
        x_test (np.ndarray): Scaled test features.
        y_test (pd.Series): Test labels.

    Returns:
        pd.DataFrame: A dataframe containing Accuracy, Recall, Precision, and F1 scores.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(x_test)
        # FLAML predict returns numpy array, ensure it matches y_test type for metrics if needed
        
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall (Sensitivity)": recall_score(y_test, y_pred, pos_label=0), 
            "Precision": precision_score(y_test, y_pred, pos_label=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=0),
        }
        results.append(metrics)
    return pd.DataFrame(results)

def main():
    """
    Main execution pipeline:
    1. Load Data
    2. Split & Scale
    3. Train Multiple AutoML Models
    4. Evaluate & Save Bundle
    """
    warnings.filterwarnings("ignore")
    print(f"🚀 Starting {APP_TITLE} Model Training Engine")
    print(f"⚙️  Time Budget: {TIME_BUDGET}s per model")

    # 1. Load Data
    df = load_data()
    
    # 2. Prepare Features/Target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 3. Split Data
    print(f"✂️  Splitting data (Test Size: {TEST_SIZE})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )

    # 4. Scale Data
    # Note: We keep StandardScaler to maintain compatibility with app.py's expected bundle structure
    # and because some algorithms (like lrl1) benefit from it.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X.columns.tolist()

    # 5. Train Models (Ensemble Strategy)
    # We train specific variations to power the "Consensus" feature in the dashboard.
    model_types = {
        "AutoML Best (LGBM)": "lgbm",
        "AutoML XGBoost": "xgboost",
        "AutoML Random Forest": "rf",
        "AutoML Extra Trees": "extra_tree",
        "AutoML Logistic Reg": "lrl1"
    }
    
    trained_models = {}
    
    print("\n🏋️  Beginning AutoML Training Loop...")
    for display_name, estimator_key in model_types.items():
        try:
            model = train_flaml_model(X_train_scaled, y_train, estimator_key, TIME_BUDGET)
            trained_models[display_name] = model
        except Exception as e:
            print(f"⚠️  Failed to train {display_name}: {e}")

    # 6. Evaluate
    print("\n📊 Evaluating Performance...")
    results_df = evaluate_models(trained_models, X_test_scaled, y_test)
    print(results_df.sort_values(by="Recall (Sensitivity)", ascending=False).to_string(index=False))

    # 7. Save Bundle
    bundle = {
        "models": trained_models,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": {
            "title": APP_TITLE,
            "class_labels": CLASS_LABELS,
            "target_column": TARGET_COLUMN
        }
    }
    joblib.dump(bundle, "models_bundle.pkl")
    print("\n📦 Model bundle saved to 'models_bundle.pkl'")
    print("✅ Ready for app.py!")

if __name__ == "__main__":
    main()
