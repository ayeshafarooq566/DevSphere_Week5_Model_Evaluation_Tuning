import os
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def tune_best_model(
    best_model_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n==========================================")
    print("MODEL TUNING")
    print("==========================================")

    # ==========================================
    # LOGISTIC REGRESSION
    # ==========================================

    if best_model_name == "Logistic Regression":

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ])

        parameter_grid = {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["liblinear", "lbfgs"]
        }

    # ==========================================
    # RANDOM FOREST
    # ==========================================

    elif best_model_name == "Random Forest":

        model = RandomForestClassifier(
            random_state=42
        )

        parameter_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5]
        }

    # ==========================================
    # SVM
    # ==========================================

    else:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=42
            ))
        ])

        parameter_grid = {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["linear", "rbf"]
        }

    # ==========================================
    # GRID SEARCH
    # ==========================================

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    print("\nRunning GridSearchCV...")
    print("Please wait...")

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation F1 Score:")
    print(f"{grid_search.best_score_:.4f}")


    # ==========================================
    # TUNING RESULTS
    # ==========================================

    tuning_results = pd.DataFrame(
        grid_search.cv_results_
    )

    os.makedirs("outputs", exist_ok=True)

    tuning_results.to_csv(
        "outputs/tuning_grid_results.csv",
        index=False
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        tuning_results
    )