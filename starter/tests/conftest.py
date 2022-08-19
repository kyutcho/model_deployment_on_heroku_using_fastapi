from lib2to3.pgen2.pgen import DFAState
import os
import pytest
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

root_dir = Path(__file__).parent.parent.resolve()

@pytest.fixture(scope="session")
def data():
    try:
        file_name = "census.csv"
        path = os.path.join(str(root_dir), "data", file_name)
        df = pd.read_csv(path)
        return df

    except FileNotFoundError as e:
        logging.error(f"File {file_name} not found")
    
    