import os
import logging
import json
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

root_dir = Path(__file__).parent.parent.parent.resolve()

def test_file_exists():
    pickle_file_list = ["model.pkl", "encoder.pkl", "labelizer.pkl"]
    for f in pickle_file_list:
        file_path = Path(os.path.join(root_dir, "starter", "model", f))

        try:
            assert file_path.is_file()
        except AssertionError as e:
            logger.error(f"File {f} does not exist")


def test_model_score_valid(model_scores):
    for k, v in model_scores.items():
            for cls in v:
                for k2, v2 in cls.items():
                    for v3 in v2.values():
                        try:
                            assert v3 >= 0.0 and v3 <= 1.0
                        except AssertionError as e:
                            logger.error("Score is outside the boundary")