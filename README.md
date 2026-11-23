# Bank Loan Approval Prediction using Machine Learning

## Project Overview

This project uses Machine Learning to predict historical bank loan approval outcomes based on applicant financial and other relevant information.

The main objective is to understand the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and model saving.

## Dataset

- **Dataset:** Loan Prediction Dataset
- **Total Records:** 614
- **Target Variable:** `Loan_Status`
- **Target Classes:**
  - `Y` – Historically approved
  - `N` – Historically rejected

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Workflow

1. Data Collection and Loading
2. Exploratory Data Analysis (EDA)
3. Handling Missing Values
4. Feature Selection
5. Encoding Categorical Features
6. Feature Scaling
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Model Comparison
11. Saving the Final Model

## Features Used

The following features were selected:

**Numerical Features**
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term

**Categorical Features**
- Education
- Credit_History
- Property_Area

The identifier `Loan_ID` and demographic attributes were excluded from the model inputs.

## Machine Learning Models

Two classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier

Logistic Regression performed slightly better on the test dataset.

## Model Performance

| Evaluation Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 86.18% | 84.55% |
| Precision | 84.00% | 82.35% |
| Recall | 98.82% | 98.82% |
| F1 Score | 90.81% | 89.84% |

### Confusion Matrix – Logistic Regression

| | Predicted Rejected | Predicted Approved |
|---|---:|---:|
| Actual Rejected | 22 | 16 |
| Actual Approved | 1 | 84 |

## Final Model

Logistic Regression was selected as the final model.

A Scikit-learn Pipeline was created to combine:
- Missing-value imputation
- Feature scaling
- Categorical encoding
- Logistic Regression classification

The trained pipeline was saved using Joblib as:

`loan_approval_model.pkl`

## Project Structure

```text
Loan_Approval_Project/
│
├── bank.py
├── README.md
├── loan_approval_model.pkl
└── train.csv
```

## Key Learnings

- Performing Exploratory Data Analysis
- Handling missing data
- Selecting relevant features
- Applying numerical scaling and categorical encoding
- Training classification models
- Evaluating models using classification metrics
- Building Scikit-learn Pipelines
- Saving trained Machine Learning models using Joblib

## Limitations

- The dataset contains historical loan decisions and may reflect historical biases.
- A high test accuracy does not guarantee reliable predictions on new data.
- Excluding demographic attributes does not guarantee fairness; other features may still act as proxies.
- This project is for educational purposes only and must not be used to make actual lending decisions.

## Conclusion

This project provided practical experience in building an end-to-end Machine Learning classification workflow.

By comparing Logistic Regression and Decision Tree models, Logistic Regression achieved the better overall performance on the selected test dataset.

---

**Project Type:** Machine Learning – Classification

**Status:** Completed
