import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel


class model_inputs(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    Usage_Frequency: int
    support_Calls: int
    Payment_Delivery: int
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: int
    Last_interaction: int


app = FastAPI()


@app.get("/Required_inputs")
def inputs(input: model_inputs):
    return (input, len(input))
