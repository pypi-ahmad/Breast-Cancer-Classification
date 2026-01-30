import warnings
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

def save_sample_data(features: pd.DataFrame, target: pd.Series) -> None:
    """Saves a small sample CSV for users to test the app."""
    sample_df = features.copy()
    sample_df["target"] = target
    sample_df.head(10).to_csv("sample_data.csv", index=False)
    print("✅ Saved 'sample_data.csv'")

def build_models(random_state: int = 42):
    """Factory function defining the model suite."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }

def evaluate_models(models, x_test, y_test):
    """Calculates metrics for all models."""
    results = []
    for name, model in models.items():
        y_pred = model.predict(x_test)
        
        # Calculate metrics (Target 1 = Malignant)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall (Sensitivity)": recall_score(y_test, y_pred, pos_label=0), # Sklearn default 0=Malignant
            "Precision": precision_score(y_test, y_pred, pos_label=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=0),
        }
        results.append(metrics)
    return pd.DataFrame(results)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. Load Data
    data = load_breast_cancer()
    # Note: In sklearn breast_cancer: 0 = Malignant, 1 = Benign
    # We will preserve this raw mapping but handle the logic in the App to be user-friendly.
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # 2. Save Sample
    save_sample_data(x, y)

    # 3. Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Scale (Crucial for SVM/KNN)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 5. Train
    models = build_models(random_state=42)
    print(f"🚀 Training {len(models)} models...")
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        print(f"   - {name} trained.")

    # 6. Evaluate
    results_df = evaluate_models(models, x_test_scaled, y_test)
    print("\n📊 Model Performance (Test Set):")
    print(results_df.sort_values(by="Recall (Sensitivity)", ascending=False).to_string(index=False))

    # 7. Save Bundle
    bundle = {
        "models": models,
        "scaler": scaler,
        "feature_names": list(x.columns),
        "target_names": ["Malignant", "Benign"] # 0, 1
    }
    joblib.dump(bundle, "models_bundle.pkl")
    print("\n📦 Saved 'models_bundle.pkl'")

if __name__ == "__main__":
    main()