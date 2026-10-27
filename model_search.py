"""Choose the model, lambda and decision threshold by 5-fold cross-validation.

Compares L2 logistic regression with and without age interactions, for each
lambda, on the validation F1 score (threshold fitted on the training folds).
For the best configuration, picks the threshold that maximizes F1 on all
out-of-fold predictions. Copy the printed LAMBDA and THRESHOLD into run.py.

Usage:
    python model_search.py
"""

import os

import numpy as np

from cross_validation import add_intercept, best_threshold, build_k_indices, f1_score
from features import fit_transform_interactions, transform_interactions
from model import logistic_newton, sigmoid
from preprocess import fit_transform, load_column_names, transform
from utils import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

LAMBDAS = [1e-2, 1e-3, 1e-4]
K_FOLD = 5
SEED = 1


def design_matrices(x_tr, x_val, names, interactions):
    """Preprocess (fitted on x_tr), optionally add age interactions, add the intercept."""
    tx_tr, features, prep = fit_transform(x_tr, names)
    tx_val, _ = transform(x_val, prep)
    if interactions:
        tx_tr, _, stats = fit_transform_interactions(tx_tr, features)
        tx_val = transform_interactions(tx_val, features, stats)
    return add_intercept(tx_tr), add_intercept(tx_val)


def main():
    x, _, y_train, _, _ = load_data(DATA_DIR)
    y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}
    names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))
    k_indices = build_k_indices(len(y), K_FOLD, SEED)

    keys = [
        (interactions, lambda_) for interactions in (False, True) for lambda_ in LAMBDAS
    ]
    f1_val = {key: [] for key in keys}
    oof = {key: np.zeros(len(y)) for key in keys}  # out-of-fold probabilities

    for k in range(K_FOLD):
        val = k_indices[k]
        tr = np.concatenate([idx for i, idx in enumerate(k_indices) if i != k])
        for interactions in (False, True):
            tx_tr, tx_val = design_matrices(x[tr], x[val], names, interactions)
            for lambda_ in LAMBDAS:
                w = logistic_newton(y[tr], tx_tr, lambda_)
                p_tr, p_val = sigmoid(tx_tr @ w), sigmoid(tx_val @ w)
                t = best_threshold(y[tr], p_tr)
                f1_val[(interactions, lambda_)].append(f1_score(y[val], p_val >= t))
                oof[(interactions, lambda_)][val] = p_val
            del tx_tr, tx_val
        print(f"fold {k + 1}/{K_FOLD} done")

    print(f"\n{'age interactions':>16} {'lambda':>7}   F1 val (mean ± std over folds)")
    for interactions, lambda_ in keys:
        f = np.array(f1_val[(interactions, lambda_)])
        print(
            f"{str(interactions):>16} {lambda_:7.0e}   {f.mean():.4f} ± {f.std():.4f}"
        )

    best = max(keys, key=lambda key: np.mean(f1_val[key]))
    threshold = best_threshold(y, oof[best], n_grid=199)
    print(f"\nBest: age interactions = {best[0]}, LAMBDA = {best[1]:.0e}")
    print(
        f"THRESHOLD = {threshold:.4f}  (out-of-fold F1 = {f1_score(y, oof[best] >= threshold):.4f})"
    )


if __name__ == "__main__":
    main()
