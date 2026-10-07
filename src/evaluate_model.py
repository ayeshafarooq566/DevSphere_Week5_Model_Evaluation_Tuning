import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


def generate_evaluation_outputs(
    model,
    X_test,
    y_test,
    results_df,
    tuning_results,
    best_model_name,
    best_params
):

    os.makedirs("outputs", exist_ok=True)


    # ==========================================
    # PREDICTIONS
    # ==========================================

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]


    # ==========================================
    # FINAL METRICS
    # ==========================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # ==========================================
    # PRINT FINAL RESULTS
    # ==========================================

    print("\n==========================================")
    print("OPTIMIZED MODEL RESULTS")
    print("==========================================")

    print(f"Model    : {best_model_name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nBest Parameters:")
    print(best_params)


    # ==========================================
    # SAVE PREDICTIONS
    # ==========================================

    predictions_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions,
        "Probability": probabilities
    })

    predictions_df.to_csv(
        "outputs/predictions.csv",
        index=False
    )


    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

    report = classification_report(
        y_test,
        predictions
    )

    with open(
        "outputs/classification_report.txt",
        "w"
    ) as file:

        file.write("MODEL EVALUATION REPORT\n")
        file.write("=======================\n\n")

        file.write(
            f"Optimized Model: {best_model_name}\n\n"
        )

        file.write(
            f"Best Parameters:\n{best_params}\n\n"
        )

        file.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall   : {recall:.4f}\n"
        )

        file.write(
            f"F1 Score : {f1:.4f}\n"
        )

        file.write(
            f"ROC-AUC  : {roc_auc:.4f}\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(report)


    # ==========================================
    # MODEL COMPARISON GRAPH
    # ==========================================

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    results_plot = results_df.set_index(
        "Model"
    )[metrics]

    ax = results_plot.plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_title(
        "Machine Learning Model Comparison"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Model"
    )

    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.legend(
        title="Metrics"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/model_comparison.png",
        dpi=300
    )

    plt.close()


    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Malignant",
            "Benign"
        ],
        yticklabels=[
            "Malignant",
            "Benign"
        ]
    )

    plt.title(
        f"Confusion Matrix - {best_model_name}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300
    )

    plt.close()


    # ==========================================
    # ROC CURVE
    # ==========================================

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"ROC Curve - {best_model_name}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/roc_curve.png",
        dpi=300
    )

    plt.close()


    # ==========================================
    # TUNING RESULTS GRAPH
    # ==========================================

    tuning_plot = tuning_results[
        ["mean_test_score"]
    ].sort_values(
        "mean_test_score",
        ascending=False
    ).head(10)

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        range(1, len(tuning_plot) + 1),
        tuning_plot["mean_test_score"],
        marker="o"
    )

    plt.title(
        "Top Hyperparameter Tuning Results"
    )

    plt.xlabel(
        "Parameter Combination Rank"
    )

    plt.ylabel(
        "Cross-Validation F1 Score"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/tuning_results.png",
        dpi=300
    )

    plt.close()


    print("\nAll evaluation graphs and reports generated successfully.")