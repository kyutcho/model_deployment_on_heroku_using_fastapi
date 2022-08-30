import pytest
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_data_shape(data):
    """
    test if data is big enough to continue
    """
    try:
        assert data.shape[0] > 1000 and data.shape[1] > 10
    except AssertionError as e:
        logger.error(f"Data shape is {data.shape} and not appropriate for ML project")

def test_data_nulls(data):
    """
    test if data has no null values
    """
    try:
        assert data.isnull().sum().sum() == 0
    except AssertionError as e:
        logger.error("Data is not imputed. Not valid for some ML algorithms")

