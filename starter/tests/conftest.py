from lib2to3.pgen2.pgen import DFAState
import os
import pytest
from pathlib import Path
import logging
import json
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

root_dir = Path(__file__).parent.parent.parent.resolve()

@pytest.fixture(scope="session")
def data():
    try:
        file_name = "census.csv"
        path = os.path.join(str(root_dir), "starter", "data", file_name)
        df = pd.read_csv(path)
        return df

    except FileNotFoundError as e:
        logger.error(f"File {file_name} not found")
    

@pytest.fixture(scope="session")
def model_scores():
    try:
        json_path = Path(os.path.join(root_dir, "starter", "outputs", "data_slice_output.json"))
        with open(json_path, 'r') as j:
            data_slice_score_dict = json.loads(j.read())

        return data_slice_score_dict

    except FileNotFoundError as e:
        logger.error("JSON score file not exists")