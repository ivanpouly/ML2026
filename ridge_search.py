import os

import matplotlib.pyplot as plt
import numpy as np

from cross_validation import iterate_folds, rmse
from implementations import ridge_regression
from preprocess import load_column_names
from utils import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FIG_PATH = os.path.join(ROOT, "figures", "ridge_lambda.png")

LAMBDAS = np.logspace(-8, 1, 19)
K_FOLD = 5
SEED = 1

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


def run_search(x, y, names, lambdas=LAMBDAS, k_fold=K_FOLD, seed=SEED):
    """Cross-validated RMSE of ridge regression for every lambda.

    Args:
        x: shape=(N, D). Raw features.
        y: shape=(N,). Labels in {0, 1}.
        names: list of D column names.
        lambdas: array of regularization strengths to try.
        k_fold: int. Number of folds.
        seed: int. Random seed for the split.

    Returns:
        rmse_train, rmse_val: arrays of shape (k_fold, len(lambdas)).
    """
    rmse_train = np.zeros((k_fold, len(lambdas)))
    rmse_val = np.zeros((k_fold, len(lambdas)))
    folds = iterate_folds(x, y, names, k_fold, seed)
    for k, (tx_tr, y_tr, tx_val, y_val) in enumerate(folds):
        for i, lambda_ in enumerate(lambdas):
            w, _ = ridge_regression(y_tr, tx_tr, lambda_)
            rmse_train[k, i] = rmse(y_tr, tx_tr @ w)
            rmse_val[k, i] = rmse(y_val, tx_val @ w)
        print(f"fold {k + 1}/{k_fold} done")
    return rmse_train, rmse_val


def plot_results(rmse_train, rmse_val, lambdas, path):
    """Train and validation RMSE against lambda: mean over folds, +/- 1 std band."""
    best = lambdas[np.argmin(rmse_val.mean(axis=0))]
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for values, color, label in [
        (rmse_train, TRAIN_COLOR, "train"),
        (rmse_val, VAL_COLOR, "validation"),
    ]:
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(lambdas, mean, color=color, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(lambdas, mean - std, mean + std, color=color, alpha=0.15, lw=0)

    ax.axvline(best, color="0.55", ls="--", lw=1, label=f"best λ = {best:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Ridge regression, {rmse_val.shape[0]}-fold cross-validation")
    ax.grid(True, color="0.9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    x_train, _, y_train, _, _ = load_data(DATA_DIR)
    y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}
    names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))

    rmse_train, rmse_val = run_search(x_train, y, names)

    mean_tr, mean_val = rmse_train.mean(axis=0), rmse_val.mean(axis=0)
    print(f"\n{'lambda':>8} {'RMSE train':>11} {'RMSE val':>9}")
    for lambda_, tr, val in zip(LAMBDAS, mean_tr, mean_val):
        print(f"{lambda_:8.0e} {tr:11.4f} {val:9.4f}")
    best = int(np.argmin(mean_val))
    print(
        f"\nBest lambda (validation RMSE): {LAMBDAS[best]:.0e}, RMSE = {mean_val[best]:.4f}"
    )

    plot_results(rmse_train, rmse_val, LAMBDAS, FIG_PATH)
    print(f"Figure saved to {FIG_PATH}")


if __name__ == "__main__":
    main()
