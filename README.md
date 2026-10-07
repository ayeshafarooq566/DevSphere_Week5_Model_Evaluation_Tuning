# DevSphere Week 5 - Model Evaluation and Tuning

## Project Title

Breast Cancer Classification - Model Comparison and Hyperparameter Tuning

## Objective

The objective of this project is to train multiple machine learning classification models, compare their performance using evaluation metrics, and optimize the best-performing model using hyperparameter tuning.

## Dataset

The Breast Cancer Wisconsin dataset provided by scikit-learn was used.

The dataset contains:

- 569 samples
- 30 numerical features
- 2 target classes

No external dataset download is required.

## Machine Learning Models

The following models were trained:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Model Tuning

GridSearchCV with 5-fold cross-validation was used to optimize the best-performing model.

## Visualizations

The project generates:

- Model comparison chart
- Confusion matrix
- ROC curve
- Hyperparameter tuning results

## Project Structure

```text
DevSphere_Week5_Model_Tuning/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── train_models.py
│   ├── tune_model.py
│   └── evaluate_model.py
│
├── models/
├── outputs/
└── screenshots/