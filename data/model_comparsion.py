import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv('data/customer.csv')
X = df.drop('Churn', axis = 1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)
logistic_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter = 1000))
])

tree_model = DecisionTreeClassifier(random_state = 42)

forest_model = RandomForestClassifier(n_estimators = 100,random_state = 42)

models = {

    "Logistic Regression": logistic_model,

    "Decision Tree": tree_model,

    "Random Forest": forest_model

}

results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    results.append(

        {

            "Model": name,

            "Accuracy":
            accuracy_score(
                y_test,
                prediction
            ),


            "Precision":
            precision_score(
                y_test,
                prediction
            ),


            "Recall":
            recall_score(
                y_test,
                prediction
            ),


            "F1":
            f1_score(
                y_test,
                prediction
            )

        }

    )



results_df = pd.DataFrame(results)


print(results_df)

for name, model in models.items():
    print(f"{name} - Training Accuracy: {model.score(X_train, y_train)}")
    print(f"{name} - Testing Accuracy: {model.score(X_test, y_test)}")

score = cross_val_score(

    RandomForestClassifier(),

    X,

    y,

    cv=5

)


print(score)

print(score.mean())

forest_model.fit(X_train, y_train)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': forest_model.feature_importances_
}
)
print(
    importance.sort_values(by='Importance', ascending=False)
)