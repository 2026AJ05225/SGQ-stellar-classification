"""Shared constants and the metric helper used by train.py and app.py."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 14 input features kept from the 17 raw feature columns.
# Dropped: obj_ID, spec_obj_ID (pure identifiers) and rerun_ID (constant = 301).
FEATURES = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "run_ID", "cam_col", "field_ID", "plate", "MJD", "fiber_ID",
]
TARGET = "class"
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC Score"]


def compute_metrics(y_true, y_pred, y_proba):
    """Six required metrics. Multi-class => macro averages, AUC is one-vs-rest."""
    try:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=list(range(len(CLASS_NAMES))),
        )
    except ValueError:  # e.g. a class is missing from the uploaded file
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }
