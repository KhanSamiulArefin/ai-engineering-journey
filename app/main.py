from fastapi import FastAPI
import joblib
import pandas as pd

from app.schemas import Customer


app = FastAPI(
    title="Customer Churn Prediction API",
    description="This API predicts customer churn based on input features.",
    version="1.0"
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


    return {
    "customer": customer.model_dump(),
    "prediction": result,
    "churn_probability": round(float(churn_probability), 2)
}