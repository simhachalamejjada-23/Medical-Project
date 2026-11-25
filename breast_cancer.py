
# ============================================================
# BREAST CANCER CLASSIFICATION USING MACHINE LEARNING
# MCA FINAL YEAR PROJECT
# ============================================================

import os
import warnings
import joblib

import pandas as pd
import numpy as np
import matplotlib

# Non-interactive backend to prevent graph display errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

print("\n==========================================")
print(" BREAST CANCER CLASSIFICATION PROJECT")
print("==========================================")

# Get the directory containing this Python file

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# YOUR ACTUAL DATASET LOCATION
# BreastCancer_ML_Project/archive/data.csv

DATASET_PATH = os.path.join(
    BASE_DIR,
    "archive",
    "data.csv"
)

# Results folder

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# Folder for saving trained machine learning models
MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.isfile(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found: {DATASET_PATH}\n"
        "Check whether archive/data.csv exists."
    )

data = pd.read_csv(
    DATASET_PATH
)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")

print(data.shape)

print("\nFirst Five Records:")

print(data.head())


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

print("\n==========================================")
print(" DATA PREPROCESSING")
print("==========================================")

# Remove unnecessary columns

data.drop(
    columns=[
        "id",
        "Unnamed: 32"
    ],
    errors="ignore",
    inplace=True
)

# Remove duplicate records

data.drop_duplicates(
    inplace=True
)

# Check diagnosis column

if "diagnosis" not in data.columns:

    raise ValueError(
        "Diagnosis column is missing from dataset."
    )

# Convert diagnosis values
# M = Malignant = 1
# B = Benign = 0

data["diagnosis"] = data["diagnosis"].map({

    "M": 1,

    "B": 0

})

# Remove rows with invalid diagnosis

data.dropna(
    subset=["diagnosis"],
    inplace=True
)

# Separate input features and target

X = data.drop(
    columns=["diagnosis"]
)

y = data["diagnosis"].astype(int)

# Convert all feature columns to numeric

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Remove rows containing missing or invalid features

valid_rows = X.notna().all(
    axis=1
)

X = X.loc[
    valid_rows
]

y = y.loc[
    valid_rows
]

# Reset indexes

X.reset_index(
    drop=True,
    inplace=True
)

y.reset_index(
    drop=True,
    inplace=True
)

print("\nPreprocessing completed!")

print("\nFeature Shape:")

print(X.shape)

print("\nTarget Shape:")

print(y.shape)

print("\nClass Distribution:")

print(
    y.value_counts().rename(
        index={
            0: "Benign",
            1: "Malignant"
        }
    )
)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

print("\n==========================================")
print(" TRAIN-TEST SPLIT")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)

print("\nTraining Samples:")

print(len(X_train))

print("\nTesting Samples:")

print(len(X_test))


# ============================================================
# 5. DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\n==========================================")
print(" MACHINE LEARNING MODELS")
print("==========================================")

models = {

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    "Logistic Regression": {

        "pipeline": Pipeline([

            ("scaler", StandardScaler()),

            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )

        ]),

        "params": {

            "model__C": [
                0.1,
                1,
                10
            ]

        }

    },


    # --------------------------------------------------------
    # SUPPORT VECTOR MACHINE
    # --------------------------------------------------------

    "SVM": {

        "pipeline": Pipeline([

            ("scaler", StandardScaler()),

            (
                "model",
                SVC(
                    probability=True,
                    random_state=RANDOM_STATE
                )
            )

        ]),

        "params": {

            "model__C": [
                0.1,
                1,
                10
            ],

            "model__kernel": [
                "linear",
                "rbf"
            ]

        }

    },


    # --------------------------------------------------------
    # K-NEAREST NEIGHBORS
    # --------------------------------------------------------

    "KNN": {

        "pipeline": Pipeline([

            ("scaler", StandardScaler()),

            (
                "model",
                KNeighborsClassifier()
            )

        ]),

        "params": {

            "model__n_neighbors": [
                3,
                5,
                7
            ],

            "model__weights": [
                "uniform",
                "distance"
            ]

        }

    },


    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    "Decision Tree": {

        "pipeline": Pipeline([

            (
                "model",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                )
            )

        ]),

        "params": {

            "model__max_depth": [
                3,
                5,
                None
            ],

            "model__min_samples_split": [
                2,
                5
            ]

        }

    },


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "Random Forest": {

        "pipeline": Pipeline([

            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE
                )
            )

        ]),

        "params": {

            "model__n_estimators": [
                100,
                200
            ],

            "model__max_depth": [
                None,
                5,
                10
            ],

            "model__min_samples_split": [
                2,
                5
            ]

        }

    },


    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    "Gradient Boosting": {

        "pipeline": Pipeline([

            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                )
            )

        ]),

        "params": {

            "model__n_estimators": [
                100,
                200
            ],

            "model__learning_rate": [
                0.05,
                0.1
            ],

            "model__max_depth": [
                2,
                3
            ]

        }

    }

}


# ============================================================
# 6. CROSS VALIDATION
# ============================================================

print("\n==========================================")
print(" CROSS VALIDATION")
print("==========================================")

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# 7. SPECIFICITY FUNCTION
# ============================================================

def calculate_specificity(
    y_true,
    y_pred
):

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]

    )

    tn = cm[0, 0]

    fp = cm[0, 1]

    if tn + fp == 0:

        return 0.0

    specificity = tn / (tn + fp)

    return specificity


# ============================================================
# 8. MODEL TRAINING AND EVALUATION
# ============================================================

print("\n==========================================")
print(" MODEL TRAINING AND EVALUATION")
print("==========================================")

results = []

trained_models = {}

best_accuracy = -1

best_model_name = None

best_model = None


for model_name, model_info in models.items():

    print("\n")

    print("==========================================")

    print("Training Model:", model_name)

    print("==========================================")

    # Pipeline

    pipeline = model_info["pipeline"]

    # Hyperparameter combinations

    parameters = model_info["params"]

    # Grid search

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=parameters,

        cv=cv,

        scoring="accuracy",

        n_jobs=-1,

        refit=True

    )

    # Train model

    grid_search.fit(

        X_train,

        y_train

    )

    # Best model

    best_estimator = grid_search.best_estimator_

    trained_models[
        model_name
    ] = best_estimator

    # Cross validation score

    cv_accuracy = grid_search.best_score_

    # Predict test data

    y_pred = best_estimator.predict(

        X_test

    )

    # Predicted probabilities

    y_prob = best_estimator.predict_proba(

        X_test

    )[:, 1]

    # Confusion matrix

    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=[0, 1]

    )

    tn = cm[0, 0]

    fp = cm[0, 1]

    fn = cm[1, 0]

    tp = cm[1, 1]

    # Accuracy

    accuracy = accuracy_score(

        y_test,

        y_pred

    )

    # Precision

    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0

    )

    # Recall

    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0

    )

    # Specificity

    specificity = calculate_specificity(

        y_test,

        y_pred

    )

    # F1 score

    f1 = f1_score(

        y_test,

        y_pred,

        zero_division=0

    )

    # ROC AUC

    auc = roc_auc_score(

        y_test,

        y_prob

    )

    # --------------------------------------------------------
    # PRINT MODEL RESULTS
    # --------------------------------------------------------

    print("\nBest Parameters:")

    print(
        grid_search.best_params_
    )

    print(
        "\nCross Validation Accuracy: "
        f"{cv_accuracy * 100:.2f}%"
    )

    print(
        "\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nPrecision: "
        f"{precision * 100:.2f}%"
    )

    print(
        "\nMalignant Recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        "\nSpecificity: "
        f"{specificity * 100:.2f}%"
    )

    print(
        "\nF1 Score: "
        f"{f1 * 100:.2f}%"
    )

    print(
        "\nROC AUC: "
        f"{auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(

            y_test,

            y_pred,

            labels=[0, 1],

            target_names=[
                "Benign",
                "Malignant"
            ],

            zero_division=0

        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Best Parameters": str(
            grid_search.best_params_
        ),

        "CV Accuracy (%)": round(
            cv_accuracy * 100,
            2
        ),

        "Test Accuracy (%)": round(
            accuracy * 100,
            2
        ),

        "Precision (%)": round(
            precision * 100,
            2
        ),

        "Malignant Recall (%)": round(
            recall * 100,
            2
        ),

        "Specificity (%)": round(
            specificity * 100,
            2
        ),

        "F1-Score (%)": round(
            f1 * 100,
            2
        ),

        "ROC-AUC": round(
            auc,
            4
        ),

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp

    })

    # --------------------------------------------------------
    # FIND BEST MODEL
    # --------------------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model_name = model_name

        best_model = best_estimator


# ============================================================
# 9. SAVE TRAINED MACHINE LEARNING MODELS
# ============================================================

print("\n==========================================")
print(" SAVING TRAINED MODELS")
print("==========================================")

# Save each fitted pipeline. Saving the complete pipeline preserves
# preprocessing steps such as StandardScaler where they are used.
for model_name, trained_model in trained_models.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_model.pkl"
    )

    model_path = os.path.join(
        MODEL_FOLDER,
        filename
    )

    joblib.dump(
        trained_model,
        model_path
    )

    print("Model saved:", model_path)

print("\nAll trained models saved successfully!")


# ============================================================
# 10. SAVE MODEL COMPARISON CSV
# ============================================================

print("\n==========================================")
print(" SAVING MODEL COMPARISON")
print("==========================================")

results_df = pd.DataFrame(

    results

)

# Sort by test accuracy

results_df.sort_values(

    by="Test Accuracy (%)",

    ascending=False,

    inplace=True

)

# CSV path

results_csv_path = os.path.join(

    OUTPUT_FOLDER,

    "model_comparison_results.csv"

)

# Save results

results_df.to_csv(

    results_csv_path,

    index=False

)

print("\nMODEL COMPARISON RESULTS:")

print(

    results_df.to_string(

        index=False

    )

)

print(

    "\nResults saved to:",

    results_csv_path

)


# ============================================================
# 11. CONFUSION MATRIX GRAPH
# ============================================================

print("\n==========================================")
print(" CONFUSION MATRIX")
print("==========================================")

# Predictions from best model

best_predictions = best_model.predict(

    X_test

)

# Confusion matrix

best_cm = confusion_matrix(

    y_test,

    best_predictions,

    labels=[0, 1]

)

# Create graph

plt.figure(

    figsize=(7, 5)

)

plt.imshow(

    best_cm,

    interpolation="nearest"

)

plt.title(

    f"Confusion Matrix - {best_model_name}"

)

plt.xlabel(

    "Predicted Label"

)

plt.ylabel(

    "Actual Label"

)

plt.xticks(

    [0, 1],

    ["Benign", "Malignant"]

)

plt.yticks(

    [0, 1],

    ["Benign", "Malignant"]

)

# Add numbers inside confusion matrix

for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            str(best_cm[i, j]),

            ha="center",

            va="center",

            fontsize=14

        )

plt.tight_layout()

# Save graph

confusion_path = os.path.join(

    OUTPUT_FOLDER,

    "confusion_matrix.png"

)

plt.savefig(

    confusion_path,

    dpi=300,

    bbox_inches="tight"

)

# Close graph

plt.close()

print(

    "\nConfusion matrix saved:",

    confusion_path

)


# ============================================================
# 12. MODEL ACCURACY COMPARISON GRAPH
# ============================================================

print("\n==========================================")
print(" MODEL ACCURACY COMPARISON")
print("==========================================")

plt.figure(

    figsize=(10, 6)

)

# Bar graph

plt.bar(

    results_df["Model"],

    results_df["Test Accuracy (%)"]

)

plt.title(

    "Machine Learning Model Accuracy Comparison"

)

plt.xlabel(

    "Machine Learning Models"

)

plt.ylabel(

    "Test Accuracy (%)"

)

plt.xticks(

    rotation=30,

    ha="right"

)

plt.ylim(

    0,

    100

)

plt.tight_layout()

# Save graph

accuracy_graph_path = os.path.join(

    OUTPUT_FOLDER,

    "model_accuracy_comparison.png"

)

plt.savefig(

    accuracy_graph_path,

    dpi=300,

    bbox_inches="tight"

)

# Close graph instead of showing it

plt.close()

print(

    "\nAccuracy graph saved:",

    accuracy_graph_path

)


# ============================================================
# 13. FINAL PROJECT SUMMARY
# ============================================================

print("\n==========================================")
print(" FINAL PROJECT SUMMARY")
print("==========================================")

print(

    "\nBest Model:",

    best_model_name

)

print(

    "Best Model Test Accuracy:",

    f"{best_accuracy * 100:.2f}%"

)

print(

    "\nTotal Models Evaluated:",

    len(models)

)

print("\nGenerated Files:")

print(
    "1. models/ (six saved trained model pipelines)"
)

print(
    "2. results/model_comparison_results.csv"
)

print(

    "3. results/confusion_matrix.png"

)

print(

    "4. results/model_accuracy_comparison.png"

)

print("\n==========================================")

print(

    " PROJECT EXECUTION COMPLETED SUCCESSFULLY!"

)

print("==========================================")
