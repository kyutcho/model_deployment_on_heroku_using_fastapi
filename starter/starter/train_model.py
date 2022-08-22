# Script to train machine learning model.

# Add the necessary imports for the starter code.
import logging
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data, remove_space_in_data, remove_space_in_column_names
from ml.model import train_model, compute_model_metrics, inference, save_model, compute_data_slice_scores

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Add code to load in the data.
root_dir = Path(__file__).parent.parent.resolve()
data_path = os.path.join(str(root_dir), "data", "census.csv")
data = pd.read_csv(data_path)

# Remove leading spaces in columns
data = remove_space_in_column_names(data)

# Remove leading spaces in category type data
data = remove_space_in_data(data, data.select_dtypes('object'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, oh_encoder, lbl_binarizer = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=oh_encoder, lb=lbl_binarizer)

model_name = "model.pkl"
try:
    model_path = os.path.join(str(root_dir), "model", model_name)
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
except FileNotFoundError as e:
    logger.info("No existing model. It will fit and save a new model")
    
    # Train and save a model.
    rf_model = train_model(X_train, y_train)

    # Save model
    save_model(rf_model)

# Get prediction using trained model
y_pred = inference(rf_model, X_test)

# Get model scores (precision, recall, fbeta)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# Get model scores for categorical slices
slice_scores = compute_data_slice_scores(test, y_test, y_pred)

with open(os.path.join(str(root_dir), "output", "slice_output.txt"), 'w') as f:
    for k, v in slice_scores.items():
        for cls in v:
            for k2, v2 in cls.items():
                f.write(f"class {k2} of categorical feature {k} has score {v2}")
