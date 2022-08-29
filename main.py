import os
import sys
from typing import Union
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()

# port = int(os.environ.get('PORT', 5000))

root_dir = Path(__file__).parent.resolve()
# sys.path.append(str(root_dir))

with open("starter/model/model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open(os.path.join(root_dir, "starter", "model", "encoder.pkl"), "rb") as f:
    oh_encoder = pickle.load(f)

with open(os.path.join(root_dir, "starter", "model", "labelizer.pkl"), "rb") as f:
    labelizer = pickle.load(f)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]

class census_data_input(BaseModel):
    age: int
    workclass: object
    fnlgt: int
    education: object
    education_num: int = Field(alias="education-num")
    marital_status: object = Field(alias="marital-status")
    occupation: object
    relationship: object
    race: object
    sex: object
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: object = Field(alias="native-country")
    
@app.get("/")
def home():
    return {"Message": "Hello World"}

@app.get("/welcome")
def get_name(name: str):
    return {"Welcome message": f"{name}"}

@app.post("/predictions")
async def predict_salary(input_param: census_data_input):
    input_data = input_param.json()
    input_dict= json.loads(input_data)

    age = input_dict['age']
    workclass = input_dict['workclass']
    fnlgt = input_dict['fnlgt']
    education = input_dict['education']
    education_num = input_dict['education_num']
    marital_status = input_dict['marital_status']
    occupation = input_dict['occupation']
    relationship = input_dict['relationship']
    race = input_dict['race']
    sex = input_dict['sex']
    capital_gain = input_dict['capital_gain']
    capital_loss = input_dict['capital_loss']
    hours_per_week = input_dict['hours_per_week']
    native_country = input_dict['native_country']

    input_df = pd.DataFrame(np.array([[age, workclass, fnlgt, education, education_num, \
                                       marital_status, occupation, relationship, race, \
                                       sex, capital_gain, capital_loss, hours_per_week, native_country]]),
                            columns=['age', 'workclass', 'fnlgt', 'education', 'education-num', \
                                     'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

    test_X_val, test_y_val, _, _ = process_data(input_df, 
                                                categorical_features=cat_features, 
                                                label=None, 
                                                training=False, 
                                                encoder=oh_encoder, 
                                                lb=labelizer)

    y_pred = inference(classifier, test_X_val)

    salary_prediction = labelizer.inverse_transform(y_pred)[0]

    return {"Prediction": salary_prediction}