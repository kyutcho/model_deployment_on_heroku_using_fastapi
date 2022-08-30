import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(root_dir))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_path():
    r = client.get("/")
    assert r.status_code == 200


def test_get_path_query():
    r = client.get("/welcome?name=John")
    assert r.status_code == 200
    assert r.json() == {"Welcome message": "John"}


def test_first_inference_case():
    '''
    Check the case where predicted salary is <=50K
    '''
    body = {"age": 31,
            "workclass": "Private",
            "fnlgt": 154374,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Machine-op-inspct",
            "relationship": "Unmarried",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"}

    r = client.post("/predictions", json=body)

    assert r.status_code == 200
    assert r.json() == {"Prediction": "<=50K"}


def test_second_inference_case():
    '''
    Check the case where predicted salary is <=50K
    '''
    body = {"age": 52,
            "workclass": "Private",
            "fnlgt": 2368,
            "education": "Doctorate",
            "education-num": 16,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 1050,
            "capital-loss": 30,
            "hours-per-week": 40,
            "native-country": "United-States"}

    r = client.post("/predictions", json=body)

    assert r.status_code == 200
    assert r.json() == {"Prediction": ">50K"}