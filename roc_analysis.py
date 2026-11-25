
# ============================================================
# BREAST CANCER CLASSIFICATION - ROC AND AUC ANALYSIS
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "archive", "data.csv"
)

MODEL_FOLDER = os.path.join(BASE_DIR, "models")

RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("\n==========================================")
print(" BREAST CANCER ROC AND AUC ANALYSIS")
print("==========================================")

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

data = pd.read_csv(DATASET_PATH)

data.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore",
    inplace=True
)

data.drop_duplicates(inplace=True)

data["diagnosis"] = data["diagnosis"].map({
    "M": 1,
    "B": 0
})

data.dropna(subset=["diagnosis"], inplace=True)

X = data.drop(columns=["diagnosis"])
y = data["diagnosis"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce")

valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset loaded successfully!")
print("Test samples:", len(X_test))

# ------------------------------------------------------------
# 3. LOAD TRAINED MODELS
# ------------------------------------------------------------

model_files = {
    "Logistic Regression":
        "logistic_regression_model.pkl",

    "SVM":
        "svm_model.pkl",

    "KNN":
        "knn_model.pkl",

    "Decision Tree":
        "decision_tree_model.pkl",

    "Random Forest":
        "random_forest_model.pkl",

    "Gradient Boosting":
        "gradient_boosting_model.pkl"
}

# ------------------------------------------------------------
# 4. ROC CURVE ANALYSIS
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

auc_results = []

for model_name, filename in model_files.items():

    model_path = os.path.join(
        MODEL_FOLDER, filename
    )

    model = joblib.load(model_path)

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test, probabilities
    )

    auc_score = roc_auc_score(
        y_test, probabilities
    )

    auc_results.append({
        "Model": model_name,
        "AUC": round(auc_score, 4)
    })

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_score:.4f})"
    )

    print(
        f"{model_name}: AUC = {auc_score:.4f}"
    )

# ------------------------------------------------------------
# 5. PLOT ROC CURVES
# ------------------------------------------------------------

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison - Breast Cancer Classification"
)

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "roc_curve_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 6. SAVE AUC COMPARISON
# ------------------------------------------------------------

auc_df = pd.DataFrame(auc_results)

auc_df.sort_values(
    by="AUC",
    ascending=False,
    inplace=True
)

auc_df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "auc_comparison.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 7. FINAL RESULTS
# ------------------------------------------------------------

print("\n==========================================")
print(" AUC COMPARISON RESULTS")
print("==========================================")

print(auc_df.to_string(index=False))

print("\nGenerated Files:")
print("1. results/roc_curve_comparison.png")
print("2. results/auc_comparison.csv")

print("\nROC AND AUC ANALYSIS COMPLETED!")
