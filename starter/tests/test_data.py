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
    assert data.shape[0] > 1000 and data.shape[1] > 10

