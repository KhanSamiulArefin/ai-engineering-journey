from pydantic import BaseModel


class Customer(BaseModel):

    Age: int

    Income: float

    Purchase: float

    Visits: int