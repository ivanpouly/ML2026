import numpy as np

from preprocess import fit_transform, transform


def build_k_indices(n, k_fold, seed=1):
    """Randomly split the indices 0..n-1 into k_fold folds of (almost) equal size.

    Args:
        n: int. Number of samples.
        k_fold: int. Number of folds.
        seed: int. Random seed, so that every run uses the same folds.

    Returns:
        list of k_fold arrays of indices.
    """
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k_fold)


def add_intercept(tx):
    """Prepend a column of ones (bias term). Must come after standardization."""
    n = tx.shape[0]
    return np.hstack([np.ones((n, 1)), tx])


def iterate_folds(x, y, names, k_fold=5, seed=1):
    """Yield the preprocessed data of each fold, one fold at a time.

    For fold k, the preprocessing is fitted on the other k_fold - 1 folds and
    applied to fold k. Preprocessing does not depend on the model's
    hyperparameters, so do it once per fold and loop over them inside.

    Args:
        x: shape=(N, D). Raw features.
        y: shape=(N,). Labels in {0, 1}.
        names: list of D column names.
        k_fold: int. Number of folds.
        seed: int. Random seed for the split.

    Yields:
        tx_tr: shape=(N_tr, F+1). Training folds, preprocessed, with intercept.
        y_tr: shape=(N_tr,). Training labels.
        tx_val: shape=(N_val, F+1). Validation fold, preprocessed with the
            training-fold statistics, with intercept.
        y_val: shape=(N_val,). Validation labels.
    """
    n = len(y)
    k_indices = build_k_indices(n, k_fold, seed)
    for k in range(k_fold):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([idx for i, idx in enumerate(k_indices) if i != k])
        tx_tr, _, prep = fit_transform(x[tr_idx], names)
        tx_val, _ = transform(x[val_idx], prep)
        yield add_intercept(tx_tr), y[tr_idx], add_intercept(tx_val), y[val_idx]


def rmse(y, scores):
    """Root mean squared error between labels and predicted scores."""
    return np.sqrt(np.mean((y - scores) ** 2))


def f1_score(y, y_pred):
    """F1 score of predictions (booleans or {0, 1}) against labels in {0, 1}."""
    y = np.asarray(y) == 1
    y_pred = np.asarray(y_pred).astype(bool)
    tp = np.sum(y & y_pred)
    fp = np.sum(~y & y_pred)
    fn = np.sum(y & ~y_pred)
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y, scores, n_grid=99):
    """Decision threshold on `scores` that maximizes the F1 score on (y, scores). """
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, n_grid))
    f1s = [f1_score(y, scores >= t) for t in candidates]
    return candidates[int(np.argmax(f1s))]


def evaluate(tx_tr, y_tr, tx_val, y_val, w):
    """RMSE and F1 score of a linear model on the training and validation folds.
    Returns:
        dict with keys rmse_train, rmse_val, f1_train, f1_val, threshold.
    """
    s_tr, s_val = tx_tr @ w, tx_val @ w
    t = best_threshold(y_tr, s_tr)
    return {
        "rmse_train": rmse(y_tr, s_tr),
        "rmse_val": rmse(y_val, s_val),
        "f1_train": f1_score(y_tr, s_tr >= t),
        "f1_val": f1_score(y_val, s_val >= t),
        "threshold": t,
    }
