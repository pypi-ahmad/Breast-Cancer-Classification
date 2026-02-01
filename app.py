"""
Universal Classification AI • Command Center.

This Streamlit application serves as a dynamic dashboard for any classification model
trained by the `train_automl.py` engine. It adapts its interface (titles, inputs,
classes) based on the metadata found in `models_bundle.pkl`.
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)


# --- App Configuration ---
st.set_page_config(
    page_title="Universal Classification AI • Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Model Bundle Loader ---
@st.cache_resource
def load_bundle():
    """
    Loads the model bundle (models, scaler, metadata) from disk.
    
    Returns:
        dict: The loaded bundle dictionary.
    """
    try:
        return joblib.load("models_bundle.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model bundle not found. Run 'python train_automl.py' first to initialize the engine.")
        st.stop()


# --- Load Model Bundle ---
bundle = load_bundle()
models = bundle["models"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# --- Universal Metadata ---
# This dashboard is "Universal" because it adapts its UI based on the metadata
# stored in the model bundle. It reads the Title, Class Labels, and Feature Names
# dynamically, allowing it to serve as a frontend for ANY classification problem
# (e.g., Breast Cancer, Heart Disease, Churn) without code changes.
metadata = bundle.get("metadata", {
    "title": "Breast Cancer Classification • Command Center",
    "class_labels": {0: "Malignant", 1: "Benign"}
})
APP_TITLE = metadata["title"]
CLASS_LABELS = metadata["class_labels"]
POS_CLASS = CLASS_LABELS[0]
NEG_CLASS = CLASS_LABELS[1]

# --- PCA Helper ---
@st.cache_data
def compute_pca(x_scaled, labels=None):
    """
    Computes 2D PCA for visualization.
    
    Args:
        x_scaled (np.ndarray): Scaled features.
        labels (pd.Series, optional): True labels for coloring points.
        
    Returns:
        tuple: (pca_df, explained_variance_ratio)
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    if labels is not None:
        pca_df["Diagnosis"] = labels.map(CLASS_LABELS)
    return pca_df, pca.explained_variance_ratio_


# --- Probability Adapter ---
def get_positive_proba(model, x):
    """
    Returns the probability of the Positive Class (Label 0).
    Handles different model types (predict_proba vs decision_function).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 0]
        if proba.ndim == 1:
            return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1 / (1 + np.exp(-scores))
    preds = model.predict(x) if hasattr(model, "predict") else np.zeros(x.shape[0])
    return np.where(preds == 0, 1.0, 0.0)


def load_dataframe_from_sklearn():
    """Loads the built-in Sklearn dataset for demo purposes."""
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    return df


def load_dataframe_from_upload(uploaded_file):
    """Loads data from a user-uploaded CSV file."""
    return pd.read_csv(uploaded_file)


# --- Header ---
st.title(APP_TITLE)
st.caption(f"Safety mapping: 0 = {POS_CLASS}, 1 = {NEG_CLASS}")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Model Selection")
    available_models = list(models.keys())
    default_models = available_models[:3] if len(available_models) >= 3 else available_models
    selected_models = st.multiselect(
        "Select Models",
        options=available_models,
        default=default_models,
    )

    st.header("🎛️ Sensitivity Tuner")
    st.caption("Lower thresholds increase sensitivity (Recall) but may raise false positives.")
    malignant_threshold = st.slider(
        f"{POS_CLASS} probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help=f"If P({POS_CLASS}) ≥ threshold → predict {POS_CLASS}.",
    )
    consensus_threshold = st.slider(
        "Consensus vote threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help=f"If ({POS_CLASS} votes / selected models) ≥ threshold → consensus {POS_CLASS}.",
    )

    st.header("Data Source")
    data_mode = st.radio("Choose Input", ["Upload CSV", "Load Sample (Sklearn)"])
    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type="csv")


# --- Data Loading ---
df = None
labels = None

if data_mode == "Load Sample (Sklearn)":
    df = load_dataframe_from_sklearn()
elif uploaded_file:
    df = load_dataframe_from_upload(uploaded_file)

if df is None:
    st.info("Awaiting data upload or sample load.")
    st.stop()

# --- Label Handling (0 = Positive, 1 = Negative) ---
if "target" in df.columns:
    labels = df["target"].astype(int)
    df_features = df.drop(columns=["target"])
else:
    df_features = df

# --- Feature Alignment + Scaling ---
df_display = df_features.copy()
df_features = df_features.reindex(columns=feature_names, fill_value=0)
X_scaled = scaler.transform(df_features)
X_unscaled = scaler.inverse_transform(X_scaled)
df_unscaled = pd.DataFrame(X_unscaled, columns=feature_names, index=df_features.index)

# --- Sidebar: Symptom Predictor ---
with st.sidebar:
    st.divider()
    st.header("🩺 Symptom Predictor")
    with st.expander("Manual Prediction", expanded=False):
        st.caption("Enter patient metrics:")
        input_data = {}
        # Smart defaults from loaded dataframe
        for feature in feature_names:
            default_val = float(df_features[feature].mean()) if feature in df_features.columns else 0.0
            input_data[feature] = st.number_input(feature, value=default_val)
        
        if st.button("Predict Diagnosis"):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Use best available model (or first selected)
            predictor_name = selected_models[0] if selected_models else available_models[0]
            predictor = models[predictor_name]
            
            # Predict
            prob_pos = get_positive_proba(predictor, input_scaled)[0]
            pred_label = POS_CLASS if prob_pos >= malignant_threshold else NEG_CLASS
            
            st.write(f"**Model used:** {predictor_name}")
            if pred_label == POS_CLASS:
                st.error(f"Prediction: **{pred_label}**")
            else:
                st.success(f"Prediction: **{pred_label}**")
            st.info(f"Probability ({POS_CLASS}): {prob_pos:.2%}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 Consensus Diagnosis",
    "📈 Ranking & Performance",
    "🔬 Deep EDA",
    "🧠 Model Explainability",
    "🛠️ Model Specs",
])

# --- TAB 1: Consensus Diagnosis ---
with tab1:
    st.subheader("Consensus Diagnosis")
    if not selected_models:
        st.warning("Select at least one model to run inference.")
    else:
        model_predictions = pd.DataFrame(index=df_features.index)
        for name in selected_models:
            model = models[name]
            pos_proba = get_positive_proba(model, X_scaled)
            pred_label = np.where(pos_proba >= malignant_threshold, 0, 1)
            model_predictions[name] = np.where(pred_label == 0, POS_CLASS, NEG_CLASS)

        pos_votes = (model_predictions == POS_CLASS).sum(axis=1)

        consensus_score = pos_votes / len(selected_models)

        results_table = df_display.copy()
        results_table["Consensus Score"] = consensus_score
        results_table["Consensus Diagnosis"] = np.where(
            consensus_score >= consensus_threshold, POS_CLASS, NEG_CLASS
        )
        results_table = pd.concat([results_table, model_predictions], axis=1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(results_table))
        col2.metric(f"Flagged {POS_CLASS}", int((results_table["Consensus Diagnosis"] == POS_CLASS).sum()))
        col3.metric("Models Selected", len(selected_models))

        display_cols = ["Consensus Diagnosis", "Consensus Score"] + list(model_predictions.columns)
        st.dataframe(
            results_table[display_cols]
            .style.background_gradient(subset=["Consensus Score"], cmap="Reds"),
            width='stretch',
        )

# --- TAB 2: Ranking & Performance ---
with tab2:
    st.subheader("Ranking & Performance")
    if labels is None:
        st.info("Provide ground truth labels to compute rankings and metrics.")
    elif not selected_models:
        st.warning("Select at least one model to compare.")
    else:
        metrics_rows = []
        roc_traces = []

        y_true_plot = (labels == 0).astype(int)

        for name in selected_models:
            model = models[name]
            tuned_proba = get_positive_proba(model, X_scaled)
            preds = np.where(tuned_proba >= malignant_threshold, 0, 1)

            metrics_rows.append({
                "Model": name,
                "Accuracy": accuracy_score(labels, preds),
                "Recall (Sensitivity)": recall_score(labels, preds, pos_label=0),
                "Precision": precision_score(labels, preds, pos_label=0),
                "F1": f1_score(labels, preds, pos_label=0),
            })

            fpr, tpr, _ = roc_curve(y_true_plot, tuned_proba)
            roc_traces.append((name, fpr, tpr, auc(fpr, tpr)))

        metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
        
        # Identify Winner
        best_model = max(roc_traces, key=lambda x: x[3])
        st.success(f"🏆 The Winner is **{best_model[0]}** with AUC **{best_model[3]:.4f}**")

        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color="#d1e7dd").format("{:.2%}"),
            width='stretch',
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Metric Leaderboard")
            long_df = metrics_df.reset_index().melt(
                id_vars="Model", var_name="Metric", value_name="Score"
            )
            fig_metrics = px.bar(
                long_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
            )
            st.plotly_chart(fig_metrics, width='stretch')

        with col_b:
            st.markdown("#### ROC Curves")
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type="line",
                line=dict(dash="dash"),
                x0=0,
                x1=1,
                y0=0,
                y1=1,
            )
            for name, fpr, tpr, auc_score in roc_traces:
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{name} (AUC={auc_score:.2f})",
                    )
                )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                title=f"ROC Curves ({POS_CLASS} as Positive)",
            )
            st.plotly_chart(fig_roc, width='stretch')

        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(selected_models))
        for i, name in enumerate(selected_models):
            tuned_proba = get_positive_proba(models[name], X_scaled)
            preds = np.where(tuned_proba >= malignant_threshold, 0, 1)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=[f"Pred {POS_CLASS}", f"Pred {NEG_CLASS}"],
                    y=[f"True {POS_CLASS}", f"True {NEG_CLASS}"],
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_cm.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250)
            with cm_cols[i]:
                st.markdown(f"**{name}**")
                st.plotly_chart(fig_cm, width='stretch', key=f"cm_{name}")

# --- TAB 3: Deep EDA ---
with tab3:
    st.subheader("Deep Exploratory Data Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Diagnosis Distribution")
        if labels is None:
            st.info("No labels available for distribution plot.")
        else:
            dist_df = labels.map(CLASS_LABELS).value_counts().reset_index()
            dist_df.columns = ["Diagnosis", "Count"]
            fig_pie = px.pie(
                dist_df,
                names="Diagnosis",
                values="Count",
                color="Diagnosis",
                color_discrete_map={POS_CLASS: "red", NEG_CLASS: "green"},
            )
            st.plotly_chart(fig_pie, width='stretch')

    with col2:
        st.markdown("#### PCA Visualization (2D)")
        pca_df, variance = compute_pca(X_scaled, labels)
        if labels is not None:
            fig_pca = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Diagnosis",
                color_discrete_map={POS_CLASS: "red", NEG_CLASS: "green"},
                title=f"Explained Variance: {variance.sum():.2%}",
            )
        else:
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2")
        st.plotly_chart(fig_pca, width='stretch')

    st.markdown("#### Feature Analysis")
    feature_choice = st.selectbox("Select Feature", feature_names, index=0)

    col3, col4 = st.columns(2)
    with col3:
        if labels is None:
            fig_hist = px.histogram(df_display, x=feature_choice)
            st.plotly_chart(fig_hist, width='stretch')
        else:
            box_df = pd.concat([df_display[feature_choice], labels.rename("target")], axis=1)
            box_df["Diagnosis"] = box_df["target"].map(CLASS_LABELS)
            fig_box = px.box(
                box_df,
                x="Diagnosis",
                y=feature_choice,
                color="Diagnosis",
                color_discrete_map={POS_CLASS: "red", NEG_CLASS: "green"},
                points="all",
            )
            st.plotly_chart(fig_box, width='stretch')

    with col4:
        st.markdown("#### Top 10 Feature Correlations")
        if labels is None:
            st.info("Labels required to compute target correlation.")
        else:
            # 1. Compute correlation with Target
            corr_data = df_features.copy()
            corr_data["target"] = labels
            corr_matrix_full = corr_data.corr()
            
            # 2. Get Top 10 features strongly correlated with Diagnosis
            top_features = (
                corr_matrix_full["target"]
                .drop("target")
                .abs()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )
            
            # 3. Plot Heatmap of ONLY those top features
            top_corr_matrix = df_features[top_features].corr()
            
            fig_corr = px.imshow(
                top_corr_matrix, 
                color_continuous_scale="RdBu_r", 
                aspect="auto",
                title="Correlation (Top 10 Features)"
            )
            st.plotly_chart(fig_corr, width='stretch')

# --- TAB 4: Model Explainability ---
with tab4:
    st.subheader("Model Explainability")
    
    if not selected_models:
        st.warning("Select at least one model to enable explainability.")
    else:
        # Allow user to pick which model to explain
        model_name = st.selectbox("Select Model for SHAP Analysis", options=selected_models)
        model_obj = models[model_name]
        
        # Helper to unwrap FLAML models for SHAP
        native_model = model_obj
        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'estimator'):
            native_model = model_obj.model.estimator
        elif hasattr(model_obj, 'estimator'):
            native_model = model_obj.estimator
        
        try:
            # Attempt to create a TreeExplainer (works for XGBoost, LightGBM, RF, etc.)
            explainer = shap.TreeExplainer(native_model)
            shap_values = explainer(X_scaled)
            
            # Fix: Random Forest returns 3D structure (samples, features, classes).
            # We slice to keep only the Positive Class (index 0 for Malignant in this app).
            if shap_values.values.ndim == 3:
                shap_values = shap_values[:, :, 0]

            st.markdown(f"#### Global Importance ({model_name})")
            fig_global = plt.figure()
            shap.summary_plot(
                shap_values.values,
                df_unscaled,
                feature_names=feature_names,
                show=False,
            )
            st.pyplot(fig_global, clear_figure=True)

            st.markdown(f"#### Local Explanation ({model_name})")
            patient_row = st.selectbox(
                "Select Patient Row",
                options=list(range(len(df_unscaled))),
                index=0,
                key="shap_patient_row",
            )

            local_exp = shap.Explanation(
                values=shap_values.values[patient_row],
                base_values=shap_values.base_values[patient_row],
                data=df_unscaled.iloc[patient_row].values,
                feature_names=feature_names,
            )

            fig_local = plt.figure()
            shap.plots.waterfall(local_exp, show=False)
            st.pyplot(fig_local, clear_figure=True)
            st.caption(f"Red bars push the risk HIGHER ({POS_CLASS}), Blue bars push it LOWER ({NEG_CLASS}).")

        except Exception as e:
            st.warning(f"SHAP failed: {e}. This model might not be tree-based (e.g. SVM, Logistic Regression).")

# --- TAB 5: Model Specs ---
with tab5:
    st.subheader("🛠️ Model Specifications")
    st.caption("Hyperparameters chosen by the training engine.")
    
    if not selected_models:
        st.info("Select models in the sidebar to view their specs.")
    else:
        for name in selected_models:
            with st.expander(f"📌 {name} Parameters", expanded=False):
                model = models[name]
                # Check if it has get_params() (Scikit-Learn / XGBoost standard)
                if hasattr(model, "get_params"):
                    st.json(model.get_params())
                else:
                    st.write(model)