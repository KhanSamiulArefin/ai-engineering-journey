import pandas as pd
from sklearn.model_selection import (
  train_test_split,
  GridSearchCV
 )

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv('data/customer.csv')

X=df.drop('Churn', axis=1)
y=df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state = 42)

parameter_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    cv = 5,
    scoring = 'f1',
    n_jobs = -1
    )

grid_search.fit(X_train, y_train)
print(
    "Best parameters found: ", grid_search.best_params_
)
best_model = grid_search.best_estimator_
prediction = best_model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        prediction
    )
)