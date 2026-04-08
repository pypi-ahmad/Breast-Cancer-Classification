"""
Universal Classification Training Engine.

This script uses Microsoft FLAML to automatically train and tune multiple classification models
(LGBM, XGBoost, Random Forest, etc.) on a specified dataset. It saves the trained models,
scaler, and metadata into a pickle bundle for the dashboard app.
"""
import warnings
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from lazypredict.Supervised import LazyClassifier
    LAZYPREDICT_AVAILABLE = True
except (ImportError, RuntimeError):
    LAZYPREDICT_AVAILABLE = False

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
LAZY_TOP_N = 5   # Number of top LazyPredict models to save in bundle

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
        except Exception as exc:
            raise ValueError(f"❌ Failed to load CSV data from '{DATA_SOURCE}': {exc}") from exc

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
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    settings = {
        "time_budget": time_budget,  # total running time in seconds
        "metric": 'roc_auc',         # optimization metric
        "task": 'classification',    # task type
        "estimator_list": [estimator_name], # Restrict to specific algorithm
        "log_file_name": str(logs_dir / f"flaml_{estimator_name}.log"),
        "verbose": 0,
        "seed": 42,
    }
    
    print(f"   ⏳ Tuning {estimator_name}...")
    automl.fit(X_train=x_train, y_train=y_train, **settings)
    print(f"      - Best config: {automl.best_config}")
    print(f"      - Best ROC-AUC: {1 - automl.best_loss:.4f}")
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
            "Recall (Sensitivity)": recall_score(y_test, y_pred, pos_label=0, zero_division=0), 
            "Precision": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
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
    warnings.filterwarnings("default")
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

    if not trained_models:
        raise RuntimeError("No FLAML models were trained successfully; aborting.")

    # 6. Evaluate FLAML models
    print("\n📊 Evaluating FLAML Performance...")
    flaml_results_df = evaluate_models(trained_models, X_test_scaled, y_test)
    flaml_results_df["Framework"] = "FLAML"
    print(flaml_results_df.sort_values(by="Recall (Sensitivity)", ascending=False).to_string(index=False))

    # 7. LazyPredict — Full Integration
    # LazyPredict trains ~30 classifiers in one call. We save the top N into the
    # bundle alongside FLAML models so they are available in the dashboard.
    # LazyPredict's internal Pipeline uses StandardScaler fit on the same X_train,
    # so the classifier weights are compatible with our bundle scaler.
    lazy_benchmark_df = pd.DataFrame()
    lazy_trained_models = {}

    if LAZYPREDICT_AVAILABLE:
        print(f"\n🔮 Running LazyPredict (all classifiers, saving top {LAZY_TOP_N})...")
        try:
            lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            # Pass UNSCALED data — LazyPredict applies its own StandardScaler pipeline
            lazy_benchmark_df, _ = lazy_clf.fit(X_train, X_test, y_train, y_test)
            print("\n📊 LazyPredict Full Benchmark:")
            print(lazy_benchmark_df.to_string())

            # Extract trained model objects from LazyPredict internals
            stored = getattr(lazy_clf, 'models_', getattr(lazy_clf, 'models', {}))

            if stored:
                top_names = lazy_benchmark_df.head(LAZY_TOP_N).index.tolist()
                for name in top_names:
                    if name in stored:
                        pipe = stored[name]
                        # Extract classifier from pipeline (skip StandardScaler step)
                        # Pipeline.steps is [(name, estimator), ...]
                        if hasattr(pipe, 'steps'):
                            clf = pipe.steps[-1][1]
                        else:
                            clf = pipe
                        display_name = f"LP: {name}"
                        lazy_trained_models[display_name] = clf
                print(f"✅ Extracted top {len(lazy_trained_models)} LazyPredict models for bundle")
            else:
                print("⚠️  Could not extract model objects from LazyPredict (version may not support it)")
        except Exception as e:
            print(f"⚠️  LazyPredict failed: {e}")
    else:
        print("\n⚠️  LazyPredict not installed – skipping. pip install lazypredict to enable.")

    # Evaluate saved LazyPredict models with our scaler (validates compatibility)
    lazy_results_df = pd.DataFrame()
    if lazy_trained_models:
        print("\n📊 Evaluating LazyPredict (saved models)...")
        lazy_results_df = evaluate_models(lazy_trained_models, X_test_scaled, y_test)
        lazy_results_df["Framework"] = "LazyPredict"
        print(lazy_results_df.sort_values(by="Recall (Sensitivity)", ascending=False).to_string(index=False))

    # Merge all models for the bundle
    all_models = {}
    all_models.update(trained_models)       # FLAML models
    all_models.update(lazy_trained_models)  # LazyPredict models

    # 8. FLAML vs LazyPredict Comparison
    print("\n" + "=" * 80)
    print("🏆 FLAML vs LazyPredict — Head-to-Head Comparison")
    print("=" * 80)

    if not lazy_results_df.empty:
        combined_df = pd.concat([flaml_results_df, lazy_results_df], ignore_index=True)
        combined_df = combined_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        print(combined_df.to_string(index=False))

        # Summary: best from each framework
        best_flaml = flaml_results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
        best_lazy = lazy_results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
        print("\n--- Best per Framework ---")
        print(f"  FLAML       : {best_flaml['Model']} (Acc={best_flaml['Accuracy']:.4f}, F1={best_flaml['F1 Score']:.4f})")
        print(f"  LazyPredict : {best_lazy['Model']} (Acc={best_lazy['Accuracy']:.4f}, F1={best_lazy['F1 Score']:.4f})")

        winner_framework = "FLAML" if best_flaml["Accuracy"] >= best_lazy["Accuracy"] else "LazyPredict"
        print(f"\n  🥇 Overall winner by Accuracy: {winner_framework}")
    else:
        print("  (LazyPredict results unavailable — showing FLAML only)")
        print(flaml_results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))

    print("=" * 80)

    # 9. Save Bundle (all models: FLAML + LazyPredict)
    bundle = {
        "models": all_models,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": {
            "title": APP_TITLE,
            "class_labels": CLASS_LABELS,
            "target_column": TARGET_COLUMN
        }
    }
    joblib.dump(bundle, "models_bundle.pkl")
    print(f"\n📦 Model bundle saved to 'models_bundle.pkl' ({len(all_models)} models: "
          f"{len(trained_models)} FLAML + {len(lazy_trained_models)} LazyPredict)")
    print("✅ Ready for app.py!")

if __name__ == "__main__":
    main()
