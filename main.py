import os
import sys
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tune_model import tune_best_model
from evaluate_model import generate_evaluation_outputs


# ==========================================
# CREATE DIRECTORIES
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading Breast Cancer dataset...")

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ==========================================
# DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
}


# ==========================================
# TRAIN AND COMPARE MODELS
# ==========================================

results = []

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ==========================================
# SAVE COMPARISON RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved.")


# ==========================================
# FIND BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(f"Best model based on F1 Score: {best_model_name}")


# ==========================================
# TUNE BEST MODEL
# ==========================================

best_model, best_params, tuned_results = tune_best_model(
    best_model_name,
    X_train,
    y_train,
    X_test,
    y_test
)


# ==========================================
# SAVE OPTIMIZED MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/optimized_model.pkl"
)

print("\nOptimized model saved to:")
print("models/optimized_model.pkl")


# ==========================================
# GENERATE FINAL EVALUATION
# ==========================================

generate_evaluation_outputs(
    best_model,
    X_test,
    y_test,
    results_df,
    tuned_results,
    best_model_name,
    best_params
)


print("\n==========================================")
print("WEEK 5 TASK COMPLETED")
print("==========================================")

print("\nGenerated files:")
print("- outputs/model_comparison.csv")
print("- outputs/predictions.csv")
print("- outputs/classification_report.txt")
print("- outputs/model_comparison.png")
print("- outputs/confusion_matrix.png")
print("- outputs/roc_curve.png")
print("- outputs/tuning_results.png")
print("- models/optimized_model.pkl")