import json
from logging import FileHandler
from collections import defaultdict
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.array, y_train: np.array):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(curr_path, "model_config.json")) as f:
        model_config = json.load(f)

    # rf_model = RandomForestRegressor(X_train, y_train, **model_config)
    rf_model = RandomForestClassifier()

    rf_model.fit(X_train, y_train)

    return rf_model


def save_model(model):
    """
    Saves model in the specified directory
    """
    root_dir = Path(__file__).parent.parent.parent.resolve()
    fileHandler = open(os.path.join(root_dir, "model", "model.pkl"), "wb")

    pickle.dump(model, fileHandler)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)

    return y_pred


def compute_data_slice_scores(test_data, y_test, y_pred):
    """ Computes model score for each data slice
    
    Inputs
    ------
    X_test: Test X data
    y_test: True y labels
    y_pred: Predicted y labels

    Returns
    -------
    cat_slice_score_dict: Model scores for each class in each categorical features
    """
    cat_features_list = test_data.select_dtypes("object").columns.to_list()
    cat_features_list.remove('salary')

    cat_slice_score_dict = defaultdict(list)

    test_data_copy = test_data.copy().reset_index(drop=True)

    for cat_feature in cat_features_list:
        for cls in test_data_copy[cat_feature].unique():
            # Get the index for each class
            cls_idx = test_data_copy[test_data_copy[cat_feature] == cls].index

            # Get three scores for each class using compute_model_metrics function
            cls_score_dict = {}

            precision, recall, fbeta = compute_model_metrics(y_test[cls_idx], y_pred[cls_idx])
            cls_score_dict[cls] = {"precision": precision, "recall": recall, "fbeta": fbeta}

            # Append it to the cat_slice_score_dict            
            cat_slice_score_dict[cat_feature].append(cls_score_dict)

    return cat_slice_score_dict


def save_data_slice_scores(score_slice_dict, output_path):
    """ Saves the score slice to the output_path in txt format
    
    Inputs
    ------
    score_slice: Dictionary of scores for each class in each feature
    output_path: Path to save the text file

    Returns
    -------
    None
    """
    with open(output_path, 'w') as f:
        for k, v in score_slice_dict.items():
            for cls in v:
                for k2, v2 in cls.items():
                    f.write(f"class {k2} of categorical feature {k} has score {v2}")