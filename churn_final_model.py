import joblib
import os
import pandas as pd
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


if os.path.exists("Final_model.pkl"):
    destree_model = joblib.load("Final_model.pkl")

    @app.post("/Predictions for the customer churn model")
    def predition(inputs: model_inputs):
        features = pd.DataFrame([inputs.model_dump()])
        model_pred = int(destree_model.predict(features))

        return {
            "Prediction": "The Customer will Churn" if model_pred == 1 else "The customer will stay"
        }

else:
    raise FileNotFoundError("The file is not available")
