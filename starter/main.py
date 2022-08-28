import os
from typing import Union
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
import json
import numpy as np
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()

model_path = os.path.join("model")

model_handler = open(os.path.join(model_path, "model.pkl"), "rb")
classifier = pickle.load(model_handler)

encoder_handler = open(os.path.join(model_path, "encoder.pkl"), "rb")
oh_encoder = pickle.load(model_handler)

labelizer_handler = open(os.path.join(model_path, "labelizer.pkl"), "rb")
labelizer = pickle.load(model_handler)

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
    hours_per_week: object = Field(alias="hours-per-week")
    native_country: object = Field(alias="native-country")
    
@app.get("/")
def home():
    return {"Message": "Hello World"}

@app.get("/welcome")
def get_name(name: str):
    return {"Welcome message": "f'{name}'"}

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
                                       sex, capital_gain, capital_loss, hours_per_week, native_country]]))

    test_X_val, test_y_val, _, _ = process_data(input_df, 
                                                categorical_features=cat_features, 
                                                label=None, 
                                                training=False, 
                                                encoder=oh_encoder, 
                                                lb=labelizer)

    y_pred = inference(classifier, test_X_val)

    return labelizer.inverse_transform(y_pred)[0]


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8080)