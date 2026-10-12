from app.models import Prediction


def create_prediction(
    db,
    customer,
    result,
    probability
):

    prediction = Prediction(

        age=customer.Age,

        income=customer.Income,

        purchase=customer.Purchase,

        visits=customer.Visits,

        prediction=result,

        probability=probability

    )


    db.add(prediction)

    db.commit()

    db.refresh(prediction)


    return prediction