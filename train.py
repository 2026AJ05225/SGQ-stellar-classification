#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Train the five classifiers on SDSS17 and save models, test data and metrics.

[star_classification.csv]
"""
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sdss_utils import (CLASS_NAMES, CLASS_TO_ID, FEATURES, METRIC_COLS,
                        MODEL_FILES, TARGET, compute_metrics)

SEED = 42
SAMPLE_SIZE = 30000   # stratified subsample (SVM is slow on 100k rows); None = use all
TEST_FRACTION = 0.20

dataset_name = "Stellar Classification Dataset - SDSS17"
dataset_source = "Kaggle (fedesoriano) / Sloan Digital Sky Survey DR17"
data_url = ("https://raw.githubusercontent.com/FentanesOscar/StellarClassification/main/star_classification.csv")
local_candidates = ["star_classification.csv", "stellar_classification.csv", "stellar_data.csv"]
if len(sys.argv) > 1:                     # optional explicit path takes priority
    local_candidates.insert(0, sys.argv[1])

# ---------------------------------------------------------------- data
df = None
for local_path in local_candidates:
    # Use a CSV already downloaded into the same folder, if present.
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"Loaded local dataset: {local_path}")
        break

if df is None:
    # No local file: fall back to the public mirror URL.
    try:
        df = pd.read_csv(data_url)
        print("Dataset loaded successfully..✅")
    except Exception as exc:
        raise FileNotFoundError(
            "Dataset file not found. Download 'star_classification.csv' from the "
            "Kaggle SDSS17 dataset and place it in the same folder as this script."
        ) from exc

# Dataset information
n_samples = int(df.shape[0])
n_features = int(df.shape[1] - 1)          # 17 feature columns + 1 target
problem_type = "multi-class classification"
print(f"{dataset_name} | {dataset_source}")
print(f"{n_samples} samples, {n_features} raw feature columns, {problem_type}")

# Known bad record: one object has u/g/z = -9999 (sensor error sentinel)
df = df[(df[["u", "g", "z"]] > -1000).all(axis=1)].dropna(subset=FEATURES + [TARGET])
print("After cleaning:", len(df), "rows")
print(df[TARGET].value_counts())

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df[TARGET],
                            random_state=SEED)

X = df[FEATURES]
y = df[TARGET].map(CLASS_TO_ID)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED)

# test_data.csv: features + original string label (this is what the app uploads)
test_out = X_test.copy()
test_out[TARGET] = [CLASS_NAMES[i] for i in y_test]
test_out.to_csv("test_data.csv", index=False)
print(f"Train: {len(X_train)} | Test: {len(X_test)} -> test_data.csv written")

# ---------------------------------------------------------------- models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=SEED), ensemble=False)
    ),
}

os.makedirs("model", exist_ok=True)
rows = {}
for name, est in models.items():
    t0 = time.time()
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    proba = est.predict_proba(X_test)
    rows[name] = compute_metrics(y_test, pred, proba)
    joblib.dump(est, os.path.join("model", MODEL_FILES[name]), compress=3)
    print(f"{name:24s} acc={rows[name]['Accuracy']:.4f}  ({time.time() - t0:.1f}s)")


results = pd.DataFrame(rows).T[METRIC_COLS].round(4)
results.index.name = "ML Model Name"
results.to_csv("model/metrics.csv")


lines = ["| ML Model Name | " + " | ".join(METRIC_COLS) ]
for name, r in results.iterrows():
    lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
with open("model/metrics_table.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n" + "\n".join(lines))
print("\nBest by MCC Score:", results["MCC Score"].idxmax())


# In[ ]:





# In[ ]:




