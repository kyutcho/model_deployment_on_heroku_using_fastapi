# Script to train machine learning model.

# Add the necessary imports for the starter code.
from genericpath import isfile
import logging
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data, remove_space_in_data, remove_space_in_column_names
from ml.model import train_model, compute_model_metrics, inference, save_model, compute_data_slice_scores, save_data_slice_scores

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
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

encoder_path = os.path.join(root_dir, "model", "encoder.pkl")
if not os.path.isfile(encoder_path):
    fileHandler = open(encoder_path, "wb")
    pickle.dump(oh_encoder, fileHandler)

labelizer_path = os.path.join(root_dir, "model", "labelizer.pkl")
if not os.path.isfile(labelizer_path):
    fileHandler = open(labelizer_path, "wb")
    pickle.dump(lbl_binarizer, fileHandler)

model_name = "model.pkl"
try:
    model_path = os.path.join(str(root_dir), "model", model_name)
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
except FileNotFoundError as e:
    logger.info("No existing model. It will fit and save a new model")
    
    # Train model
    rf_model = train_model(X_train, y_train)

    # Save model
    save_model(rf_model)

print(X_test.shape)
print(X_test[0, :].reshape(1, -1).shape)

# Get prediction using trained model
# y_pred = inference(rf_model, X_test)
y_pred = inference(rf_model, X_test[0, :].reshape(1, -1))

print(y_pred[0])

print(lbl_binarizer.inverse_transform(y_pred)[0])

# # Get model scores (precision, recall, fbeta)
# precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# # Get model scores for categorical slices
# data_slice_scores = compute_data_slice_scores(test, y_test, y_pred)

# # Save slice_output.txt file
# output_path = os.path.join(str(root_dir), "outputs", "slice_output.txt")
# if not os.path.isfile(output_path):
#     save_data_slice_scores(data_slice_scores, output_path)
