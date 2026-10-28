import os

import numpy as np

from cross_validation import add_intercept, f1_score
from features import fit_transform_interactions, transform_interactions
from helpers import create_csv_submission
from model import logistic_newton, sigmoid
from preprocess import fit_transform, load_column_names, transform
from utils import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SUBMISSION_PATH = os.path.join(ROOT, "submission.csv")

LAMBDA = 1e-4
THRESHOLD = 0.2046

# load data
x_train, x_test, y_train, train_ids, test_ids = load_data(DATA_DIR)
y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}
print(
    f"x_train: {x_train.shape} | x_test: {x_test.shape} | positive rate: {y.mean():.1%}"
)

# preprocess the data
names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))
tx_train, features, prep = fit_transform(x_train, names)
tx_test, _ = transform(x_test, prep)
del x_train, x_test

# add age interactions and intercept
tx_train, all_features, stats = fit_transform_interactions(tx_train, features)
tx_test = transform_interactions(tx_test, features, stats)
tx_train, tx_test = add_intercept(tx_train), add_intercept(tx_test)
print(f"tx_train: {tx_train.shape} | tx_test: {tx_test.shape}")

# train and make predictions
w = logistic_newton(y, tx_train, LAMBDA)
p_train, p_test = sigmoid(tx_train @ w), sigmoid(tx_test @ w)
print(f"train F1: {f1_score(y, p_train >= THRESHOLD):.4f}")

# create the submission file
y_pred = np.where(p_test >= THRESHOLD, 1, -1)
print(
    f"predicted positive rate: train {np.mean(p_train >= THRESHOLD):.1%} | test {np.mean(y_pred == 1):.1%}"
)
create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
print(f"Submission written to {SUBMISSION_PATH}")
