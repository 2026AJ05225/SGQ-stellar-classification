# SGQ Stellar Classification (SDSS DR17) — Star vs Galaxy vs Quasar

---

## A. Problem Statement

The objective is to build, compare, and deploy supervised machine-learning classifiers—specifically five classical ML models/classification algorithms trained on the same dataset—to predict and identify astronomical objects as a **GALAXY**, **STAR**, or **QSO** (quasar). Utilizing numerical observations comprising photometric and spectroscopic measurements from the Sloan Digital Sky Survey Data Release 17 (SDSS DR17), this three-class classification problem evaluates model performance across six comprehensive evaluation metrics: **Accuracy**, **AUC** (Area Under the Curve, Macro OvR), **Precision** (Macro), **Recall** (Macro), **F1-score** (Macro), and **Matthews Correlation Coefficient (MCC)**. Ultimately, the trained and saved model pipelines are demonstrated in an interactive, deployed Streamlit web application that accepts and evaluates an uploaded test CSV file.

---

## B. Dataset Description

* **Dataset Name & Source:** Stellar Classification Dataset — SDSS17 (by fedesoriano on Kaggle), derived from Sloan Digital Sky Survey Data Release 17 (SDSS DR17).
* **Dataset Sources & Downloads:**
  * **Kaggle Link:** [Kaggle SDSS17 Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
  * **Direct CSV Mirror:** [GitHub Raw Mirror](https://raw.githubusercontent.com/FentanesOscar/StellarClassification/main/star_classification.csv)
  * **Expected Local Path:** `dataset/star_classification.csv` (or `Downloads/star_classification.csv`)
* **Domain:** Astronomy / Astrophysics.
* **Task & Target:** Three-class supervised multi-class classification predicting target `class` $\in$ {**GALAXY**, **QSO**, **STAR**}.
* **Dataset Size & Structure:** Contains 100,000 raw observations across 18 total columns (17 numeric input/feature columns + 1 target column `class`). This comfortably satisfies the assignment requirement of at least 12 features and 500 instances.
* **Class Balance (Cleaned Data — 99,999 rows):**
  * **GALAXY:** 59,445 instances (~59.4%)
  * **STAR:** 21,593 instances (~21.6%)
  * **QSO:** 18,961 instances (~19.0%)

*Note:* Because the classes are imbalanced, macro-averaged Precision, Recall, and F1-score are reported alongside MCC.

### Input Features & Data Dictionary

| Feature | Type / Category | Meaning & Physical Context | Included in Training |
|---|---|---|---|
| `alpha`, `delta` | Celestial Coordinates | Right ascension ($\alpha$) and declination ($\delta$) representing sky position | Yes |
| `u`, `g`, `r`, `i`, `z` | Photometry | Photometric magnitudes in ultraviolet ($u$), green ($g$), red ($r$), near-IR ($i$), and IR ($z$) filter bands | Yes |
| `redshift` | Spectroscopy | Spectroscopic redshift measurement derived from spectral lines | Yes |
| `run_ID`, `cam_col`, `field_ID` | Survey Metadata | Imaging run number, camera column, and field identifier | Yes |
| `plate`, `MJD`, `fiber_ID` | Spectroscopic Metadata | Spectroscopic plate number, Modified Julian Date (observation date), and fiber identifier | Yes |
| `obj_ID`, `spec_obj_ID` | Unique Identifiers | Unique database object ID and spectroscopic object ID | **No (Dropped)** |
| `rerun_ID` | Survey Metadata | Image processing rerun identifier (constant value across dataset) | **No (Dropped)** |

### Feature Handling, Selection & Scientific Limitations

* **Dropped Columns:** `obj_ID` and `spec_obj_ID` were dropped as pure database identifiers, and `rerun_ID` was removed due to zero variance (constant value). The remaining 14 numerical features (exceeding the 12-feature minimum requirement) were retained for model training.
* **Methodological Limitation (Survey Identifiers):** While standard models leverage physical photometric magnitudes (`u`, `g`, `r`, `i`, `z`) and spectroscopic `redshift`, retaining observational metadata (e.g., `run_ID`, `plate`, `MJD`) can introduce survey-specific spatial or instrumental artifacts. High performance on this dataset may not guarantee generalization to data from other telescopes or sky surveys. A feature-ablation experiment isolating purely physical attributes is recommended for future scientific validation.

### Data Cleaning, Sampling & Preprocessing

1. **Data Cleaning:** Removed 1 anomalous record containing sentinel noise values (-9999 in the `u`, `g`, and `z` bands).
2. **Sampling & Stratified Split:** To mitigate computational bottlenecks during Support Vector Machine (SVM) training (which scales $O(n^3)$ during probability estimation), a stratified subsample of 30,000 rows was extracted.
3. **Train/Test Partition:** Split 80:20 in a stratified manner with `random_state = 42`:
   * **Train Set:** 24,000 rows (used for model fitting)
   * **Test Set:** 6,000 rows (held-out test split, saved as `test_data.csv`)
   * **Held-out Test Class Distribution:** 3,567 GALAXY, 1,138 QSO, 1,295 STAR.
4. **Feature Scaling:** `StandardScaler` was incorporated inside `scikit-learn` pipelines for distance- and gradient-sensitive algorithms (Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machines). Tree-based models (Decision Tree) bypass scaling and operate on raw feature values.

---

## C. GitHub Repository Structure & Links

* **GitHub Repository URL:** [https://github.com/2026AJ05225/SGQ-stellar-classification/tree/main](https://github.com/2026AJ05225/SGQ-stellar-classification/tree/main)
* **Live Streamlit Web Application:** [https://sgq-stellar-classification.streamlit.app/](https://sgq-stellar-classification.streamlit.app/)

### Directory Layout

```text
SGQ-stellar-classification/
│
├── app.py                  # Main Streamlit web application interface
├── sdss_utils.py           # Helper utilities for metrics, feature list, and class mapping
├── train.ipynb             # Jupyter Notebook for EDA, training, validation & model export
├── train.py / train_models.py # Standalone Python script for automated CLI training
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── test_data.csv           # Held-out test split (6,000 rows, 14 features + class)
│
├── dataset/
│   └── star_classification.csv  # Raw SDSS DR17 dataset (100,000 rows)
│
└── model/                  # Exported model artifacts & generated outputs
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv         # Measured metrics output
    └── metrics_table.md    # Formatted Markdown metrics output
```

---

## D. Models Used and Evaluation

The following five classification algorithms were trained on the identical 24,000-row training set and evaluated on the exact same 6,000-row held-out test dataset:

1. **Logistic Regression:** Multiclass configuration with `max_iter=2000` inside a `StandardScaler` pipeline.
2. **Decision Tree Classifier:** Hyperparameter-tuned decision tree with `max_depth=12`.
3. **K-Nearest Neighbors (KNN):** Distance-weighted distance algorithm with $k=7$ neighbors inside a `StandardScaler` pipeline.
4. **Gaussian Naive Bayes:** Probabilistic classifier inside a `StandardScaler` pipeline.
5. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel classifier with $C=1.0$ and `probability=True` inside a `StandardScaler` pipeline.

### Model Comparison Table (6,000-Row Held-Out Test Set)

| ML Model Name | Accuracy | AUC (Macro OvR) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.9575 | **0.9876** | 0.9523 | 0.9522 | 0.9517 | 0.9250 |
| **Decision Tree** | **0.9702** | 0.9718 | **0.9671** | **0.9639** | **0.9655** | **0.9470** |
| **K-Nearest Neighbors** | 0.8907 | 0.9583 | 0.9077 | 0.8469 | 0.8727 | 0.8029 |
| **Gaussian Naive Bayes** | 0.8997 | 0.9708 | 0.8777 | 0.9144 | 0.8920 | 0.8324 |
| **Support Vector Machine** | 0.9563 | 0.9835 | 0.9542 | 0.9489 | 0.9512 | 0.9226 |

---

### Per-Class Recall & Misclassification Analysis

| ML Model Name | GALAXY Recall | QSO Recall | STAR Recall | Total Test Errors | Primary Confusion Pair |
|---|---:|---:|---:|---:|---|
| **Logistic Regression** | 96.2% | 89.5% | **100.0%** | 255 | 120 QSO misclassified as GALAXY |
| **Decision Tree** | **97.7%** | **91.8%** | 99.6% | **179** | 93 QSO misclassified as GALAXY |
| **K-Nearest Neighbors** | 96.3% | 85.4% | 72.4% | 656 | 353 STAR misclassified as GALAXY |
| **Gaussian Naive Bayes** | 87.3% | 87.9% | 99.2% | 602 | 423 GALAXY misclassified as QSO |
| **Support Vector Machine** | 96.6% | 90.9% | 97.2% | 262 | 103 QSO misclassified as GALAXY |

---

### Detailed Observations on Model Performance

* **Logistic Regression:** Demonstrates high overall performance (Accuracy: 0.9575) and achieves the **highest AUC score (0.9876)** across all models. It achieved perfect separation for stars (1,295 out of 1,295 correct). Its errors stem from distinguishing high-redshift quasars (QSO) from galaxies (120 quasars misclassified as galaxies), indicating that while the decision space is largely linearly separable after scaling, minor non-linear overlaps remain between galaxies and quasars.
* **Decision Tree Classifier:** **Top overall performer.** Achieved the highest Accuracy (0.9702), Macro F1 (0.9655), and MCC (0.9470), making only 179 errors out of 6,000 test records. It captured sharp physical thresholds (such as specific spectroscopic `redshift` boundaries). Its slightly lower AUC (0.9718) is expected due to the step-function nature of decision tree leaf probability estimates.
* **K-Nearest Neighbors (KNN):** Weakest performer (Accuracy: 0.8907, MCC: 0.8029). KNN struggled heavily on the STAR class, misclassifying 353 stars as galaxies (STAR recall: 72.4%). This degradation occurs because Euclidean distance calculation in 14-dimensional space gets influenced by non-physical survey metadata columns (`run_ID`, `MJD`, `plate`), even after standard scaling.
* **Gaussian Naive Bayes:** Achieved an Accuracy of 0.8997 but suffered from low precision on quasars. It severely over-predicted QSO by misclassifying 423 galaxies as quasars. This stems from the violation of feature independence assumptions—photometric passband magnitudes (`u`, `g`, `r`, `i`, `z`) are strongly correlated with one another. However, it proved computationally lightweight and achieved strong STAR recall (99.2%).
* **Support Vector Machine (SVM):** Showed strong performance (Accuracy: 0.9563, F1: 0.9512, AUC: 0.9835), performing almost identically to Logistic Regression. However, fitting non-linear RBF decision boundaries required significantly higher computational overhead (63.0s runtime vs 0.6s for Decision Tree) without yielding an accuracy advantage over tree-based methods.
* **Overall Winner:** **Decision Tree Classifier.** It delivers superior metric scores (MCC: 0.9470, F1: 0.9655, Accuracy: 0.9702), fewest misclassification errors (179), balanced per-class recall, and fast execution speed.

---

## E. Streamlit Application

The interactive web application (`app.py`), titled **"SGQ Sky Object Classifier"**, provides a user-friendly interface for model evaluation and inference:

1. **Interactive Data Upload & Sample Selection:** Allows users to upload any external test CSV dataset, select a pre-bundled sample test split, or download `test_data.csv` directly from the interface.
2. **Model Selection Dropdown:** Toggle dynamically between all five pre-trained model pipelines (`Logistic Regression`, `Decision Tree`, `KNN`, `Gaussian Naive Bayes`, `Support Vector Machine`).
3. **Automated Six Metric Evaluation:** Computes and displays Accuracy, AUC (Macro OvR), Precision (Macro), Recall (Macro), F1-Score (Macro), and MCC Score.
4. **Visual Confusion Matrix & Classification Report:** Renders interactive confusion matrix heatmaps using `seaborn`/`matplotlib` alongside per-class performance tables.
5. **All-Models Performance Comparison:** Interactive bar charts and summary tables comparing all 5 saved models.
6. **Batch CSV Prediction Export:** Generates predictions and probability scores for uploaded CSV test files, available for download.

---

## F. How to Run

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/2026AJ05225/SGQ-stellar-classification.git
cd SGQ-stellar-classification

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
[On Jupyter Notebook: pip install -r requirements.txt]
```

### 2. Dataset Placement

Place the downloaded `star_classification.csv` dataset in the dataset directory:
```text
dataset/star_classification.csv
```
OR
```text
Downloads/star_classification.csv
```

Also,even if we don't download and place the dataset in the directory, Model script will automatically check and download the dataset if not exist locally.

### 3. Model Training & Pipeline Serialization

Execute model training and evaluation via CLI or Jupyter Notebook:

**Option A: Running the Standalone Python Script**
```bash
python train.py --data dataset/star_classification.csv
```

**Option B: Executing the Jupyter Notebook**
```bash
jupyter notebook train.ipynb
```
Select **"Run All Cells"**. Training cleans the dataset, splits partitions (80:20), trains all 5 algorithms, writes metrics to `model/metrics.csv`, and serializes pipeline models as `.joblib` files.

#### Measured Training Execution Times (BITS Virtual Lab)
* **Logistic Regression:** ~2.2 s
* **Decision Tree Classifier:** ~0.6 s
* **K-Nearest Neighbors:** ~5.2 s
* **Gaussian Naive Bayes:** ~0.1 s
* **Support Vector Machine (RBF):** ~63.0 s

### 4. Launching the Local Streamlit App

Open and run the terminal in the directory of `app.py`.
```bash
streamlit run app.py
```
OR
Open `https://sgq-stellar-classification.streamlit.app/` in your browser.

### 5. Deployment to Streamlit Community Cloud

1. Push repository commits to GitHub (ensure `model/` directory and `test_data.csv` are tracked).
2. Sign into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app linked to repository `2026AJ05225/SGQ-stellar-classification`, branch `main`, and main file `app.py`.
4. Click **Deploy**.

---

## G. Reproducibility & Limitations

* **Deterministic Random State:** All sampling, splitting, and model initializations use `random_state = 42`.
* **Subsampling Protocol:** Subsampling to 30,000 total rows preserves computational viability for SVM probability estimation while maintaining class distribution ratios.
* **Library Version Consistency:** Ensure `scikit-learn` version parity between training environments and deployment servers to prevent serialization discrepancies during `.joblib` loading.
* **Astronomical Generalization Warning:** The inclusion of non-physical survey metadata (`plate`, `MJD`, `fiber_ID`, `run_ID`) means performance metrics reflect the SDSS DR17 observation strategy. Deployment on non-SDSS sky survey datasets (e.g., Gaia, LSST) would require retraining exclusively on physical magnitudes (`u, g, r, i, z`) and spectroscopic attributes.

