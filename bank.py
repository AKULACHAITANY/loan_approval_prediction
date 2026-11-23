import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\CHAITANYA\Downloads\train.csv")
print(df.head())
print(df.shape)
print(df.info())
missing = df.isnull().sum()

print(missing)
missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage)
print("Duplicate rows:", df.duplicated().sum())
print(df['Loan_Status'].value_counts())
print(df['Loan_Status'].value_counts(normalize=True) * 100)
categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    print("\nColumn:", col)
    print(df[col].value_counts(dropna=False))
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

print(numerical_columns)
df[numerical_columns].hist(
    figsize=(12, 8),
    bins=20
)

plt.tight_layout()
plt.show()

df = df.drop('Loan_ID', axis=1)
X = df.drop('Loan_Status', axis=1)

y = df['Loan_Status']
numerical_columns = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term'
]

categorical_columns = [
    'Education',
    'Credit_History',
    'Property_Area'
]
feature_columns = numerical_columns + categorical_columns

X = df[feature_columns].copy()

y = df['Loan_Status'].map({
    'Y': 1,
    'N': 0
})
from sklearn.impute import SimpleImputer

numerical_imputer = SimpleImputer(strategy='median')

categorical_imputer = SimpleImputer(strategy='most_frequent')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns)
])
print("Target values:")
print(df['Loan_Status'].value_counts(dropna=False))

print("Missing values in y:", y.isnull().sum())

print("Unique values:", df['Loan_Status'].unique())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Apply preprocessing

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")

print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)
print("Features used by model:", X.columns.tolist())
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_processed, y_train)
print("Logistic Regression model trained successfully!")
# Make predictions
y_pred = model.predict(X_test_processed)

print("Predictions:")
print(y_pred)

print("Actual values:")
print(y_test.values)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Rejected (0)', 'Approved (1)'],
    yticklabels=['Rejected (0)', 'Approved (1)']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)

dt_model.fit(X_train_processed, y_train)

dt_pred = dt_model.predict(X_test_processed)
print("Decision Tree Results")

print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))
print("F1 Score:", f1_score(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))
models = ['Logistic Regression', 'Decision Tree']

accuracy = [0.8618, 0.8455]
f1_scores = [0.9081, 0.8984]

x = range(len(models))

plt.figure(figsize=(8, 5))

plt.plot(models, accuracy, marker='o', label='Accuracy')
plt.plot(models, f1_scores, marker='o', label='F1 Score')

plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.legend()
plt.grid()

plt.show()
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

print("Final Model Accuracy:", accuracy_score(y_test, final_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_pred))
import joblib

joblib.dump(final_model, 'loan_approval_model.pkl')

print("Model saved successfully!")