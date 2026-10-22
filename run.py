import os

import numpy as np

from preprocess import fit_transform, load_column_names, transform
from utils import load_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def add_intercept(x):
    """Add to the left of the design matrix a column of ones for the intercept term.
    Args:
        x: shape=(N, D). The design matrix.
    Returns:
        shape=(N, D+1). The design matrix with an additional column of ones.
    """
    N = x.shape[0]
    return np.hstack([np.ones((N, 1)), x])


x_train, x_test, y_train, train_ids, test_ids = load_data(DATA_DIR)

# Convert labels from {-1, 1} to {0, 1}.
y01_train = (y_train + 1) / 2

names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))
tx_train, features, prep = fit_transform(x_train, names)
tx_test, _ = transform(x_test, prep)

tx_train = add_intercept(tx_train)
tx_test = add_intercept(tx_test)
features = ["intercept"] + features

print("tx_train:", tx_train.shape, "| tx_test:", tx_test.shape)
print("NaN:", np.isnan(tx_train).sum(), np.isnan(tx_test).sum())
print("intercept OK:", np.all(tx_train[:, 0] == 1))
print("positive rate:", y01_train.mean())
