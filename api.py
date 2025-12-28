import pandas as pd
from fastapi import FastAPI

from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
import joblib
app = FastAPI()

class HeartData(BaseModel):
    Age:int=Field(...,description='enter age',example=45)
    Sex:int=Field(...,description='enter sex (0 , 1)',example=1)
    ChestPainType:int=Field(...,description='enter chest pain (0 1 2 3)',example=2)
    RestingBP:int=Field(...,description='enter bp',example=180)
    Cholesterol:int=Field(...,description='enter Cholesterol',example=250)
    FastingBS:int=Field(...,description='enter fasting yes no (0,1)',example=1)
    RestingECG:int=Field(...,description='enter ECG (0,1,2)',example=1)
    MaxHR :int=Field(...,description='enter Max Heart rate',example=120)
    ExerciseAngina:int=Field(...,description='enter Exercise Angina (1=Yes, 0=No):',example=1)
    Oldpeak:float=Field(...,description='Change in heart function after exercise compared to rest (1.0 - 5.0):',example=2.2)
    ST_Slope:int=Field(...,description='ST Slope (0-up)(1-flat)(2-down):',example=2)
    NumMajorVessels:int=Field(...,description='Number of Major Vessels (0-3): ',example=3)
    Thalassemia:int=Field(...,description='Thalassemia (1-3):',example=2)


@app.post('/predict')
def predict_heart_decies(d:HeartData):
    model=joblib.load('Heart_Decies_predict.pkl')


    user_input = pd.DataFrame([[d.Age, d.Sex, d.ChestPainType, d.RestingBP, d.Cholesterol, d.FastingBS, d.RestingECG, d.MaxHR, d.ExerciseAngina, d.Oldpeak, d.ST_Slope, d.NumMajorVessels, d.Thalassemia]],columns=[
    "Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG",
    "MaxHR","ExerciseAngina","Oldpeak","ST_Slope","NumMajorVessels","Thalassemia"])

    result=model.predict(user_input)[0]
    output= 'fit and fine' if result==0 else 'Hostpital ja laude'

    return JSONResponse(status_code=200,content={'message':output})
