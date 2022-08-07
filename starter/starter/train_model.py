"""
Module: Script to train machine learning model.
Name: Jayden Cho
Date: Aug 2022
"""

# Add the necessary imports for the starter code.
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Add code to load in the data.
path = os.path.join("starter", "data", "census.csv")
data = pd.read_csv(path)

# Remove leading spaces in columns
new_col = [c.lstrip() for c in data.columns.to_list()]
old_col = data.columns.to_list()

col_name_dict = {}
for o, n in zip(old_col, new_col):
    col_name_dict[o] = n

data = data.rename(col_name_dict, axis=1)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
def preprocess_data(data, label):
    num_features = data.select_dtypes("number").columns.to_list()
    cat_features = data.select_dtypes("object").columns.to_list()

    # Redundant to education number
    cat_features.remove("education")

    y = data.pop(label)

    # Imputation with median for possible missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_features", num_imputer, num_features),
            ("cat_features", cat_imputer, cat_features)
        ]
    )

    return data, y ,preprocessor


def train_model(X_train, y_train, preprocessor, rf_config):
    rf_model = RandomForestRegressor(**rf_config)

    inference_pipeline = make_pipeline(
        preprocessor,
        rf_model
    )

    # Train a model.
    inference_pipeline.fit(X_train, y_train)

    # Save inference pipeline
    pipeline_path = os.path.join("starter", "model", "pipeline.pkl")
    joblib.dump(inference_pipeline, pipeline_path)


