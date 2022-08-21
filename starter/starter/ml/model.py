import json
from logging import FileHandler
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
    root_path = Path(__file__).parent.parent.parent.resolve()
    fileHandler = open(os.path.join(root_path, "model", "model.pkl"), "wb")

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


if __name__ == '__main__':
    save_model(None)