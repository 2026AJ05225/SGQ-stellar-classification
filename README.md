# Stellar Classification (SDSS17) — Star vs Galaxy vs Quasar
This project develops a three-class machine learning pipeline to classify astronomical objects from the Sloan Digital Sky Survey (SDSS Data Release 17) as a STAR, GALAXY, or QSO (quasar) using photometric and spectroscopic measurements.
## a. Problem statement

The objective is to build, compare, and deploy supervised machine-learning classifiers—specifically five classical ML models/classification algorithms trained on the same dataset—to predict and identify astronomical objects as a **GALAXY**, **STAR**, or **QSO** (quasar). Utilizing numerical observations comprising photometric and spectroscopic measurements from the Sloan Digital Sky Survey Data Release 17 (SDSS17), this three-class classification problem evaluates model performance across six comprehensive evaluation metrics: **Accuracy**, **AUC** (Area Under the Curve), **Precision**, **Recall**, **F1-score**, and **Matthews Correlation Coefficient (MCC)**. Ultimately, the trained and saved models are demonstrated in a deployed Streamlit web application that accepts and evaluates an uploaded test CSV file.

## b. Dataset Description

* **Dataset Name:** Stellar Classification Dataset — SDSS17 (by fedesoriano on Kaggle), derived from Sloan Digital Sky Survey Data Release 17 (SDSS DR17)
* **Dataset Source:**
* **URL:** 
1. https://raw.githubusercontent.com/FentanesOscar/StellarClassification/main/star_classification.csv
2. https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

* **Expected local path:** `Downloads/star_classification.csv`

* **Domain:** Astronomy / Astrophysics.
* **Task & Target:** Three-class supervised multi-class classification predicting the target `class` $\in$ {**GALAXY**, **QSO**, **STAR**}.
* **Dataset Size & Structure:** Contains 100,000 raw observations across 18 total columns (17 numeric input/feature columns + 1 target column `class`). This comfortably satisfies the baseline requirement of at least 12 features and 500 instances.
* **Class Balance (Full Data):**
* **GALAXY:** ~59,400 instances
* **STAR:** ~21,600 instances
* **QSO:** ~19,000 instances


**Note:** Due to class imbalance, macro-averaged evaluation metrics are reported.

### Input Features & Data Dictionary

| Feature | Type / Category | Meaning & Physical Context |
| --- | --- | --- |
| `alpha`, `delta` | Celestial Coordinates | Right ascension ($\alpha$) and declination ($\delta$) representing sky position |
| `u`, `g`, `r`, `i`, `z` | Photometry | Photometric magnitudes in ultraviolet, green, red, near-IR, and IR passband filters |
| `redshift` | Spectroscopy | Spectroscopic redshift measurement derived from the object's spectrum |
| `run_ID`, `cam_col`, `field_ID` | Survey Metadata | Imaging run number, camera column, and field identifier |
| `plate`, `MJD`, `fiber_ID` | Spectroscopic Metadata | Spectroscopic plate number, Modified Julian Date (observation date), and fiber identifier |
| `obj_ID`, `spec_obj_ID` | Unique Identifiers | Database object ID and spectroscopic object ID |
| `rerun_ID` | Survey Metadata | Image processing rerun identifier (constant value across dataset) |

### Feature Handling, Selection & Scientific Limitations

* **Dropped Columns:** `obj_ID` and `spec_obj_ID` are dropped as unique record identifiers, and `rerun_ID` is removed due to zero variance (constant value). The remaining 14 features are utilized for model training.
* **Methodological Limitation (Survey Identifiers):** While standard models leverage physical photometric magnitudes (`u, g, r, i, z`) and spectroscopic `redshift`, retaining observational/survey metadata (e.g., `run_ID`, `plate`, `MJD`) can introduce dataset-specific spatial/instrumental artifacts that limit broader scientific generalization. This limitation is noted for analysis, with feature-ablation experiments recommended to assess physical model robustness.

### Data Cleaning, Sampling & Preprocessing

* **Data Cleaning:** Removed 1 anomalous record containing sentinel noise values (-9999 in the `u`, `g`, and `z` bands).
* **Sampling & Stratified Split:** To mitigate computational bottlenecks (specifically $O(n^3)$ runtime scaling during SVM probability estimations), a stratified subsample of 30,000 rows is extracted. This subset is split 80/20 (stratified, `random_state = 42`) into:
* **Train Set:** 24,000 rows (used for model training and cross-validation)
* **Test Set:** 6,000 rows (saved as `test_data.csv` for independent evaluation and Streamlit app testing)


* **Feature Preprocessing:** Applied `StandardScaler` inside pipeline architectures for distance- and gradient-sensitive algorithms (Logistic Regression, K-Nearest Neighbors, Naive Bayes, and Support Vector Machines). Tree-based classifiers (Decision Tree) bypass feature scaling.
## c. GitHub repository link

<https://github.com/2026AJ05225/SGQ-stellar-classification/tree/main>

Live app: <https://sgq-stellar-classification.streamlit.app/>

```
sdss17-stellar-classifier/
|-- app.py
|-- sdss_utils.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/   (logistic_regression, decision_tree, knn, naive_bayes, svm .joblib + metrics.csv)
```

Run locally: `pip install -r requirements.txt`, then `python train_models.py star_classification.csv`, then `streamlit run app.py`.

## d. Models used

Metrics are computed on the held-out test split. Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.

<PASTE CONTENTS OF model/metrics_table.md HERE>

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <WRITE IN YOUR OWN WORDS> |
| Decision Tree | <WRITE IN YOUR OWN WORDS> |
| KNN | <WRITE IN YOUR OWN WORDS> |
| Naive Bayes | <WRITE IN YOUR OWN WORDS> |
| Support Vector Machine | <WRITE IN YOUR OWN WORDS> |
| **Overall winner for this dataset** | <MODEL NAME + one-line reason> |

## Streamlit app features

- CSV upload of test data (with a bundled-sample option)
- Model selection dropdown (five models)
- Accuracy, AUC, Precision, Recall, F1 and MCC display
- Confusion matrix heatmap and classification report
- All-models comparison table and chart, plus downloadable predictions
- Saved models are loaded from `model/`; nothing is retrained on the uploaded data
