import joblib
import numpy as np

def predict(features):
    model = joblib.load("models/model.pkl")
    prediction = model.predict([features])
    return prediction[0]
