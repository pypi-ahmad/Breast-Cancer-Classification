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
    page_title="Breast Cancer Classification • Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Model Bundle Loader ---
@st.cache_resource
def load_bundle():
    try:
        return joblib.load("models_bundle.pkl")
    except FileNotFoundError:
        st.error("Model bundle not found. Run 'train.py' first.")
        st.stop()


# --- PCA Helper ---
@st.cache_data
def compute_pca(x_scaled, labels=None):
    pca = PCA(n_components=2)
    components = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    if labels is not None:
        pca_df["Diagnosis"] = labels.map({0: "Malignant", 1: "Benign"})
    return pca_df, pca.explained_variance_ratio_


# --- Probability Adapter ---
def get_malignant_proba(model, x):
    """Return probability of Malignant (label 0 in sklearn dataset)."""
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
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    return df


def load_dataframe_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)


# --- Header ---
st.title("🏥 Breast Cancer Classification • Command Center")
st.caption("Safety mapping: 0 = Malignant, 1 = Benign")

# --- Load Model Bundle ---
bundle = load_bundle()
models = bundle["models"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

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
        "Malignant probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="If P(Malignant) ≥ threshold → predict Malignant.",
    )
    consensus_threshold = st.slider(
        "Consensus vote threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="If (Malignant votes / selected models) ≥ threshold → consensus Malignant.",
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

# --- Label Handling (0 = Malignant, 1 = Benign) ---
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

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Consensus Diagnosis",
    "📈 Ranking & Performance",
    "🔬 Deep EDA",
    "🧠 Model Explainability",
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
            malignant_proba = get_malignant_proba(model, X_scaled)
            pred_label = np.where(malignant_proba >= malignant_threshold, 0, 1)
            model_predictions[name] = np.where(pred_label == 0, "Malignant", "Benign")

        malignant_votes = (model_predictions == "Malignant").sum(axis=1)
        consensus_score = malignant_votes / len(selected_models)

        results_table = df_display.copy()
        results_table["Consensus Score"] = consensus_score
        results_table["Consensus Diagnosis"] = np.where(
            consensus_score >= consensus_threshold, "Malignant", "Benign"
        )
        results_table = pd.concat([results_table, model_predictions], axis=1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(results_table))
        col2.metric("Flagged Malignant", int((results_table["Consensus Diagnosis"] == "Malignant").sum()))
        col3.metric("Models Selected", len(selected_models))

        display_cols = ["Consensus Diagnosis", "Consensus Score"] + list(model_predictions.columns)
        st.dataframe(
            results_table[display_cols]
            .style.background_gradient(subset=["Consensus Score"], cmap="Reds"),
            use_container_width=True,
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
            tuned_proba = get_malignant_proba(model, X_scaled)
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
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color="#d1e7dd").format("{:.2%}"),
            use_container_width=True,
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
            st.plotly_chart(fig_metrics, use_container_width=True)

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
                title="ROC Curves (Malignant as Positive)",
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(selected_models))
        for i, name in enumerate(selected_models):
            tuned_proba = get_malignant_proba(models[name], X_scaled)
            preds = np.where(tuned_proba >= malignant_threshold, 0, 1)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred Malignant", "Pred Benign"],
                    y=["True Malignant", "True Benign"],
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_cm.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250)
            with cm_cols[i]:
                st.markdown(f"**{name}**")
                st.plotly_chart(fig_cm, use_container_width=True)

# --- TAB 3: Deep EDA ---
with tab3:
    st.subheader("Deep Exploratory Data Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Diagnosis Distribution")
        if labels is None:
            st.info("No labels available for distribution plot.")
        else:
            dist_df = labels.map({0: "Malignant", 1: "Benign"}).value_counts().reset_index()
            dist_df.columns = ["Diagnosis", "Count"]
            fig_pie = px.pie(
                dist_df,
                names="Diagnosis",
                values="Count",
                color="Diagnosis",
                color_discrete_map={"Malignant": "red", "Benign": "green"},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("#### PCA Visualization (2D)")
        pca_df, variance = compute_pca(X_scaled, labels)
        if labels is not None:
            fig_pca = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Diagnosis",
                color_discrete_map={"Malignant": "red", "Benign": "green"},
                title=f"Explained Variance: {variance.sum():.2%}",
            )
        else:
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2")
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("#### Feature Analysis")
    feature_choice = st.selectbox("Select Feature", feature_names, index=0)

    col3, col4 = st.columns(2)
    with col3:
        if labels is None:
            fig_hist = px.histogram(df_display, x=feature_choice)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            box_df = pd.concat([df_display[feature_choice], labels.rename("target")], axis=1)
            box_df["Diagnosis"] = box_df["target"].map({0: "Malignant", 1: "Benign"})
            fig_box = px.box(
                box_df,
                x="Diagnosis",
                y=feature_choice,
                color="Diagnosis",
                color_discrete_map={"Malignant": "red", "Benign": "green"},
                points="all",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        corr_matrix = df_features.corr()
        fig_corr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 4: Model Explainability ---
with tab4:
    st.subheader("Model Explainability (XGBoost)")
    if "XGBoost" not in selected_models:
        st.info("Select XGBoost to enable explainability. Tree-based models offer the best interpretability.")
    else:
        xgb_model = models.get("XGBoost")
        if xgb_model is None:
            st.warning("XGBoost model not found in the bundle.")
        else:
            st.markdown("#### Global Importance (SHAP Beeswarm)")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(X_scaled)

            fig_global = plt.figure()
            shap.summary_plot(
                shap_values.values,
                df_unscaled,
                feature_names=feature_names,
                show=False,
            )
            st.pyplot(fig_global, clear_figure=True)

            st.markdown("#### Local Explanation (Doctor's View)")
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
            st.caption("Red bars push the risk HIGHER (Malignant), Blue bars push it LOWER (Benign).")