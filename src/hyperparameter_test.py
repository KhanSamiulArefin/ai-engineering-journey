from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model1 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model2=RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
