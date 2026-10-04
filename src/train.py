# ==========================================
# Customer Intelligence System
# Day 2 - ML Training Pipeline
# Customer Churn Prediction
# ==========================================


# -------------------------------
# 1. Import Libraries
# -------------------------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -------------------------------
# 2. Load Dataset
# -------------------------------

print("Loading dataset...")

df = pd.read_csv("data/customer.csv")

print("\nFirst 5 rows:")
print(df.head())


# -------------------------------
# 3. Basic Data Understanding
# -------------------------------

print("\nDataset Information:")
print(df.info())


print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


# -------------------------------
# 4. Separate Features and Target
# -------------------------------

# Features (Input)
X = df.drop("Churn", axis=1)


# Target (Output)
y = df["Churn"]


print("\nFeatures:")
print(X.head())


print("\nTarget:")
print(y.head())


# -------------------------------
# 5. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -------------------------------
# 6. Create ML Pipeline
# -------------------------------

pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# -------------------------------
# 7. Train Model
# -------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)


print("Training completed!")


# -------------------------------
# 8. Model Prediction
# -------------------------------

predictions = pipeline.predict(
    X_test
)


# -------------------------------
# 9. Model Evaluation
# -------------------------------

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


print("\n========== MODEL PERFORMANCE ==========")

print(
    "Accuracy:",
    accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1 Score:",
    f1
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)



# -------------------------------
# 10. Save Trained Model
# -------------------------------

model_path = "models/churn_model.pkl"


joblib.dump(
    pipeline,
    model_path
)


print(
    f"\nModel saved successfully: {model_path}"
)



# -------------------------------
# 11. Test New Customer Prediction
# -------------------------------

print("\n========== NEW CUSTOMER TEST ==========")


new_customer = pd.DataFrame(
    [
        {
            "Age": 47,
            "Income": 85000,
            "Purchase": 900,
            "Visits": 2
        }
    ]
)


prediction = pipeline.predict(
    new_customer
)


probability = pipeline.predict_proba(
    new_customer
)


if prediction[0] == 1:

    print(
        "Customer is likely to CHURN"
    )

else:

    print(
        "Customer is likely to STAY"
    )


print(
    "Prediction Probability:",
    probability
)



# -------------------------------
# 12. Prediction Function
# -------------------------------

def predict_customer(
    age,
    income,
    purchase,
    visits
):

    customer = pd.DataFrame(
        [
            {
                "Age": age,
                "Income": income,
                "Purchase": purchase,
                "Visits": visits
            }
        ]
    )


    result = pipeline.predict(
        customer
    )


    if result[0] == 1:
        return "Customer is likely to churn"

    else:
        return "Customer is likely to stay"



# Function Test

print(
    "\nFunction Test:"
)

print(
    predict_customer(
        age=30,
        income=40000,
        purchase=300,
        visits=8
    )
)