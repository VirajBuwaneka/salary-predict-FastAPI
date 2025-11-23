from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("salary_model.joblib")

class InputData(BaseModel):
    experience: float
    age: float
    score: float

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running!"}

@app.post("/predict_salary")
def predict_salary(data: InputData):
    features = [[data.experience, data.age, data.score]]
    salary = model.predict(features)[0]
    return {"predicted_salary": salary}
