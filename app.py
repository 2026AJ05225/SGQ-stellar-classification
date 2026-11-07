"""Streamlit app: evaluate saved SDSS17 classifiers on an uploaded test CSV.
Models are loaded from model/*.joblib and are NOT retrained."""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from sdss_utils import (CLASS_NAMES, CLASS_TO_ID, FEATURES, METRIC_COLS,
                        MODEL_FILES, TARGET, compute_metrics)

st.set_page_config(page_title="SDSS17 Sky Object Classifier", page_icon="🔭",
                   layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(HERE, "model", MODEL_FILES[name]))


@st.cache_data(show_spinner="Scoring all five models...")
def score_all(X, y):
    metrics, preds, probas = {}, {}, {}
    for name in MODEL_FILES:
        mdl = load_model(name)
        preds[name] = mdl.predict(X)
        probas[name] = mdl.predict_proba(X)
        metrics[name] = compute_metrics(y, preds[name], probas[name])
    return metrics, preds, probas


# ------------------------------------------------------------ header
st.title("🔭 Star, Galaxy or Quasar?")
st.caption("SDSS DR17 stellar classification | five classical ML models | "
           "saved models, no retraining on your upload")

# ------------------------------------------------------------ sidebar
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead")

SAMPLE_PATH = os.path.join(HERE, "test_data.csv")
if os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "rb") as f:
        st.sidebar.download_button(
            "⬇️ Download sample test_data.csv", f.read(),
            file_name="test_data.csv", mime="text/csv",
            help="Download, then upload it above to try the app.")
st.sidebar.markdown("Required columns:")
st.sidebar.code(", ".join(FEATURES + [TARGET]), language=None)

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif use_sample:
    raw = pd.read_csv(os.path.join(HERE, "test_data.csv"))
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled sample) to begin.")
    ref = os.path.join(HERE, "model", "metrics.csv")
    if os.path.exists(ref):
        st.subheader("Reference results from training run")
        st.dataframe(pd.read_csv(ref).set_index("ML Model Name"),
                     use_container_width=True)
    st.stop()

# ------------------------------------------------------------ validation
missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

data = raw.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
labels = data[TARGET].astype(str).str.strip().str.upper()
unknown = sorted(set(labels) - set(CLASS_NAMES))
if unknown:
    st.error(f"Unexpected class label(s) {unknown}. Allowed: {CLASS_NAMES}")
    st.stop()

X = data[FEATURES]
y = labels.map(CLASS_TO_ID).to_numpy()

counts = labels.value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows scored", f"{len(data):,}")
for col, cls in zip((c2, c3, c4), CLASS_NAMES):
    col.metric(cls, f"{counts.get(cls, 0):,}")

metrics, preds, probas = score_all(X, y)

tab_sel, tab_all, tab_pred = st.tabs(
    [f"📊 {model_name}", "🏁 All models compared", "🧾 Predictions"])

# ------------------------------------------------------------ selected model
with tab_sel:
    m = metrics[model_name]
    cols = st.columns(len(METRIC_COLS))
    for col, key in zip(cols, METRIC_COLS):
        col.metric(key, "n/a" if pd.isna(m[key]) else f"{m[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, preds[model_name], labels=range(len(CLASS_NAMES)))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="mako_r", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(
            y, preds[model_name], labels=range(len(CLASS_NAMES)),
            target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ------------------------------------------------------------ comparison
with tab_all:
    table = pd.DataFrame(metrics).T[METRIC_COLS]
    table.index.name = "ML Model Name"
    st.subheader("Metrics on the uploaded data")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#2b6e4f"),
                 use_container_width=True)
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.success(f"Best MCC on this data: **{table['MCC'].idxmax()}**")
    st.download_button("⬇️ Download metrics table (CSV)",
                       table.round(4).to_csv().encode(),
                       file_name="model_comparison_metrics.csv", mime="text/csv")

# ------------------------------------------------------------ predictions
with tab_pred:
    out = data.copy()
    out["predicted_class"] = [CLASS_NAMES[i] for i in preds[model_name]]
    out["correct"] = out["predicted_class"] == labels
    st.write(f"Predictions from **{model_name}**")
    st.dataframe(out.head(500), use_container_width=True)
    st.download_button("Download predictions (CSV)",
                       out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_')}.csv")
