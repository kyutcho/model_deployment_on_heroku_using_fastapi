import requests
import json

get_r = requests.get("https://udacity-mlops-project-app.herokuapp.com")

print(get_r.text)

test_data_dict = {"age": 52,
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

post_r = requests.post(
    url = "https://udacity-mlops-project-app.herokuapp.com/predictions",
    data = json.dumps(test_data_dict),
    headers={"Content-Type": "application/json"},
)

print(f"status code: {post_r.status_code}")
print(f"outcome: {post_r.text}")