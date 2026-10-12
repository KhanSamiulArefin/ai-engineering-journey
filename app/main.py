from fastapi import FastAPI
import joblib
import pandas as pd

from app.schemas import Customer

from app.database import engine
from app.models import Base

app = FastAPI(
    title="Customer Churn Prediction API",
    description="This API predicts customer churn based on input features.",
    version="1.0"
)


Base.metadata.create_all(
    bind=engine
)


# Load trained model
model = joblib.load(
    "models/churn_model.pkl"
)


@app.get("/")
def home():

    return {
        "message": "Customer Intelligence API is running."
    }



@app.post("/predict")
def predict(
    customer: Customer
):

    data = pd.DataFrame(
        [customer.model_dump()]
    )


    prediction = model.predict(
        data
    )
    probability = model.predict_proba(
        data
    )
    churn_probability = probability[0][1] * 100  # Probability of churn


    if prediction[0] == 1:

        result = "Customer is likely to churn."

    else:

        result = "Customer is likely to stay."

    db = SessionLocal()


    create_prediction(
        db,
        customer,
        result,
        churn_probability
    )


    db.close()
    return {
    "customer": customer.model_dump(),
    "prediction": result,
    "churn_probability": round(float(churn_probability), 2)
}
@app.get("/model-info")
def model_info():
    return{
        'model': 'Logistic Regression Churn Model',
        'task': 'Predicts customer churn based on input features',
        'version': '1.0',
        'features': ['Age', 'Income', 'Purchase', 'Visits'],
    }