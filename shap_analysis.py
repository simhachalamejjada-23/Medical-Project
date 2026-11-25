
# ============================================================
# BREAST CANCER CLASSIFICATION - SHAP EXPLAINABLE AI
# ============================================================

import os
import warnings
import joblib
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "archive", "data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "random_forest_model.pkl"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR, "results"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("\n==========================================")
print(" BREAST CANCER SHAP ANALYSIS")
print("==========================================")

# ------------------------------------------------------------
# 2. LOAD AND PREPROCESS DATASET
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

# Recreate original train-test split
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
# 3. LOAD RANDOM FOREST MODEL
# ------------------------------------------------------------

model = joblib.load(MODEL_PATH)

rf_model = model.named_steps["model"]

print("\nRandom Forest model loaded successfully!")

# ------------------------------------------------------------
# 4. CREATE SHAP EXPLAINER
# ------------------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test)

# Extract malignant class (class 1)
if isinstance(shap_values, list):
    shap_values_malignant = shap_values[1]

elif getattr(shap_values, "ndim", 0) == 3:
    shap_values_malignant = shap_values[:, :, 1]

else:
    shap_values_malignant = shap_values

print("SHAP values calculated successfully!")

# ------------------------------------------------------------
# 5. SHAP SUMMARY PLOT
# ------------------------------------------------------------

print("\nGenerating SHAP Summary Plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values_malignant,
    X_test,
    show=False,
    max_display=15
)

plt.title(
    "SHAP Summary Plot - Breast Cancer Classification"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 6. SHAP FEATURE IMPORTANCE BAR PLOT
# ------------------------------------------------------------

print("\nGenerating SHAP Feature Importance Plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values_malignant,
    X_test,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title(
    "SHAP Feature Importance - Random Forest"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "shap_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 7. SAVE SHAP VALUES TO CSV
# ------------------------------------------------------------

shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean Absolute SHAP Value":
        abs(shap_values_malignant).mean(axis=0)
})

shap_importance.sort_values(
    by="Mean Absolute SHAP Value",
    ascending=False,
    inplace=True
)

shap_importance.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "shap_feature_importance.csv"
    ),
    index=False
)

print("\nSHAP FEATURE IMPORTANCE")
print(shap_importance.to_string(index=False))

# ------------------------------------------------------------
# 8. FINAL OUTPUT
# ------------------------------------------------------------

print("\n==========================================")
print(" SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nGenerated Files:")

print("1. results/shap_summary.png")
print("2. results/shap_feature_importance.png")
print("3. results/shap_feature_importance.csv")
