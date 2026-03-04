import sys
import os

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from fastapi import FastAPI
from pydantic import BaseModel
from  src.predict import predict

app = FastAPI()

class CustomerData(BaseModel):
    avg_session_length: float
    time_on_app: float
    time_on_website: float
    length_of_membership: float

@app.post("/predict")
def get_prediction(data: CustomerData):
    features = [data.avg_session_length, data.time_on_app, data.time_on_website, data.length_of_membership]
    result = predict(features)
    return {"Yearly Amount Spent Prediction": result}
