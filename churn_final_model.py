from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).parent / "Final_model.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
destree_model = joblib.load(MODEL_PATH)

# API field name -> column name the pipeline was trained with
COLUMN_MAP = {
    "Age": "Age",
    "Gender": "Gender",
    "Tenure": "Tenure",
    "Usage_Frequency": "Usage Frequency",
    "Support_Calls": "Support Calls",
    "Payment_Delay": "Payment Delay",
    "Subscription_Type": "Subscription Type",
    "Contract_Length": "Contract Length",
    "Total_Spend": "Total Spend",
    "Last_Interaction": "Last Interaction",
}


class ModelInputs(BaseModel):
    Age: int = Field(ge=18, le=65)
    Gender: Literal["Male", "Female"]
    Tenure: int = Field(ge=1, le=60)
    Usage_Frequency: int = Field(ge=1, le=30)
    Support_Calls: int = Field(ge=0, le=10)
    Payment_Delay: int = Field(ge=0, le=30)
    Subscription_Type: Literal["Basic", "Standard", "Premium"]
    Contract_Length: Literal["Monthly", "Quarterly", "Annual"]
    Total_Spend: float = Field(ge=100, le=1000)
    Last_Interaction: int = Field(ge=1, le=30)

    # runs before the Literal check: " annual " / "ANNUAL" to "Annual"
    @field_validator("Gender", "Subscription_Type", "Contract_Length", mode="before")
    @classmethod
    def normalise_text(cls, value):
        if isinstance(value, str):
            return value.strip().capitalize()
        return value


app = FastAPI(title="Customer Churn API")


@app.get("/required_inputs")
def required_inputs():
    return ModelInputs.model_json_schema()


@app.post("/predict")
def predict(inputs: ModelInputs):
    row = inputs.model_dump()
    features = pd.DataFrame([row]).rename(columns=COLUMN_MAP)
    prediction = int(destree_model.predict(features)[0])

    return {
        "prediction": "The customer will churn" if prediction == 1
        else "The customer will stay"
    }
