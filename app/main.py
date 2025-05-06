from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

@app.get("/")
def root():
    return {"message": "API de predicción de churn. Visita /docs para más información."}

