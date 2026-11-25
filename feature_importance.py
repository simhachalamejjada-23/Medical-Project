# ============================================================
# FEATURE IMPORTANCE ANALYSIS - RANDOM FOREST
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib

# Use a non-interactive backend so the script can save graphs reliably.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD TRAINED RANDOM FOREST PIPELINE
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained Random Forest model was not found: {MODEL_PATH}\n"
        "Run 'py breast_cancer.py' first to train and save the models."
    )

model_pipeline = joblib.load(MODEL_PATH)

# breast_cancer.py saves the complete fitted pipeline.
# Random Forest has a pipeline step named 'model'.
if not hasattr(model_pipeline, "named_steps") or "model" not in model_pipeline.named_steps:
    raise TypeError(
        "The saved file is not the expected scikit-learn pipeline "
        "with a step named 'model'."
    )

random_forest = model_pipeline.named_steps["model"]

if not hasattr(random_forest, "feature_importances_"):
    raise TypeError(
        "The loaded model does not provide feature_importances_. "
        "Make sure random_forest_model.pkl is a fitted RandomForestClassifier pipeline."
    )


# ============================================================
# 3. GET FEATURE NAMES AND IMPORTANCE SCORES
# ============================================================

if hasattr(model_pipeline, "feature_names_in_"):
    feature_names = list(model_pipeline.feature_names_in_)
else:
    # Fallback for pipelines trained without pandas feature names.
    dataset_path = os.path.join(BASE_DIR, "archive", "data.csv")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            "Could not find archive/data.csv to recover feature names."
        )

    data = pd.read_csv(dataset_path)
    data = data.drop(
        columns=["id", "Unnamed: 32", "diagnosis"],
        errors="ignore"
    )
    feature_names = list(data.columns)

importance_scores = random_forest.feature_importances_

if len(feature_names) != len(importance_scores):
    raise ValueError(
        f"Feature-name count ({len(feature_names)}) does not match "
        f"importance-score count ({len(importance_scores)}). "
        "Check that the saved model was trained with the current dataset."
    )

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 4. DISPLAY AND SAVE FEATURE IMPORTANCE RESULTS
# ============================================================

print("\n==========================================")
print(" RANDOM FOREST FEATURE IMPORTANCE")
print("==========================================")
print(feature_df.to_string(index=False))

csv_path = os.path.join(
    RESULTS_FOLDER,
    "feature_importance.csv"
)
feature_df.to_csv(csv_path, index=False)


# ============================================================
# 5. PLOT TOP 10 FEATURES
# ============================================================

top_features = feature_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features - Random Forest")
plt.tight_layout()

graph_path = os.path.join(
    RESULTS_FOLDER,
    "feature_importance.png"
)
plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nFeature importance analysis completed successfully!")
print("CSV saved to:", csv_path)
print("Graph saved to:", graph_path)
