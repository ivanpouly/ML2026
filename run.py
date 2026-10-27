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

LAMBDA = 1e-2  # chosen by 5-fold cross-validation (model_search.py)
THRESHOLD = 0.1962  # F1-optimal threshold on out-of-fold predictions (model_search.py)

x_train, x_test, y_train, train_ids, test_ids = load_data(DATA_DIR)
y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}
print(
    f"x_train: {x_train.shape} | x_test: {x_test.shape} | positive rate: {y.mean():.1%}"
)

# Column names from the CSV headers (Id removed); both files must match
names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))
if names != load_column_names(os.path.join(DATA_DIR, "x_test.csv")):
    raise ValueError("x_train.csv and x_test.csv do not have the same columns")

# Preprocessing: every statistic is fitted on the training set only
tx_train, features, prep = fit_transform(x_train, names)
tx_test, _ = transform(x_test, prep)
del x_train, x_test  # free ~1 GB before building the larger matrices

# Age interactions, then intercept
tx_train, all_features, stats = fit_transform_interactions(tx_train, features)
tx_test = transform_interactions(tx_test, features, stats)
tx_train, tx_test = add_intercept(tx_train), add_intercept(tx_test)
print(f"tx_train: {tx_train.shape} | tx_test: {tx_test.shape}")

# L2-regularized logistic regression on the whole training set
w = logistic_newton(y, tx_train, LAMBDA)
p_train, p_test = sigmoid(tx_train @ w), sigmoid(tx_test @ w)
print(f"train F1: {f1_score(y, p_train >= THRESHOLD):.4f}")

# Predictions in {-1, 1}, the format required by AIcrowd
y_pred = np.where(p_test >= THRESHOLD, 1, -1)
print(
    f"predicted positive rate: train {np.mean(p_train >= THRESHOLD):.1%} | test {np.mean(y_pred == 1):.1%}"
)
create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
print(f"Submission written to {SUBMISSION_PATH}")
