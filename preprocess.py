import os

import numpy as np


def read_column_names(data_dir):
    """Return the column names of x_train."""
    with open(os.path.join(data_dir, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]  # [1:] removes "Id"


def preprocess_column(col, invalid_codes):
    """Given a column, set Nan and the list of invalid codes to the median of the column
    and standardize the column.

    Args:
        col: shape=(N,). The column to fit.
        invalid_codes: list of values that are considered invalid.
    Returns:
        dict: A dictionary containing the median, mean,
        and standard deviation of the column.
    """
    col = col.copy()
    col[np.isin(col, invalid_codes)] = np.nan
    med = np.nanmedian(col)
    filled = np.where(np.isnan(col), med, col)
    std = filled.std()
    return {"med": med,
            "mean": filled.mean(), "std": std if std > 0 else 1.0}

def transform_column(col, p):
    """Apply train stats p to any column (train, val or test).
    Returns (standardized column, missing indicator)."""
    col = col.astype(float).copy()
    col[np.isin(col, p["invalid"])] = np.nan
    missing = np.isnan(col).astype(float)
    col[np.isnan(col)] = p["med"]
    return (col - p["mean"]) / p["std"], missing

DROP = {"FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "SEQNO", "_PSU",
        "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT", "NUMADULT",
        "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE",
        "LANDLINE", "HHADULT"




        "HAREHAB1"}  # à compléter
# _STATE = a number that represents the state of the US where the observation was made.
# DISPCODE = 1100 if complete interview, 1200 if partial complete interview

def preprocess_data(x_train, y_train, x_test, train_ids, test_ids):
    """
    Preprocess the data by normalizing the features and converting labels to one-hot encoding.

    Args:
        x_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test features.
        train_ids (numpy.ndarray): Training sample IDs.
        test_ids (numpy.ndarray): Test sample IDs.

    Returns:
        tuple: Preprocessed training features, one-hot encoded training labels, training sample IDs, and test sample IDs.
    """
    y_train = (y_train + 1) / 2

    cols = read_column_names(DATA_DIR)
    keep = np.array([c not in DROP for c in cols])  # True = colonne gardée

    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    cols = [c for c, k in zip(cols, keep) if k]  # garder les noms alignés avec les colonnes
    return x_train, y_train_one_hot, train_ids, test_ids