from fastapi import FastAPI, HTTPException

import joblib
import pandas as pd


from app.schemas import Customer

from app.database import (
    engine,
    SessionLocal
)

from app.models import Base

from app.crud import create_prediction

from app.logger import logger

from app.config import MODEL_PATH



app = FastAPI(

    title="Customer Churn Prediction API",

    description="This API predicts customer churn based on input features.",

    version="1.0"

)



# Create database tables

Base.metadata.create_all(
    bind=engine
)



# Load trained ML model

model = joblib.load(
    MODEL_PATH
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

    logger.info(
        "Prediction request received"
    )


    try:

        # Convert input into dataframe

        data = pd.DataFrame(
            [customer.model_dump()]
        )


        # Model prediction

        prediction = model.predict(
            data
        )


        probability = model.predict_proba(
            data
        )


        churn_probability = (
            probability[0][1] * 100
        )


        # Convert prediction result

        if prediction[0] == 1:

            result = "Customer is likely to churn."

        else:

            result = "Customer is likely to stay."



        logger.info(
            f"Prediction completed: {result}"
        )



        # Save prediction into database

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


            "churn_probability":
            round(
                float(churn_probability),
                2
            )

        }



    except Exception as e:


        logger.error(
            f"Prediction failed: {str(e)}"
        )


        raise HTTPException(

            status_code=500,

            detail="Prediction failed"

        )




@app.get("/model-info")

def model_info():

    return {


        "model":
        "Logistic Regression Churn Model",


        "task":
        "Predicts customer churn based on input features",


        "version":
        "1.0",


        "features":
        [
            "Age",
            "Income",
            "Purchase",
            "Visits"
        ]

    }