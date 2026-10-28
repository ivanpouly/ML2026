"""Reproduce the cross-validation numbers of Tables I and II of the report.

All experiments use the same 5 folds (seed 1) and report the F1 score of the
positive class: mean +/- std over folds, and paired differences between two
models as mean +/- standard error over folds.

- Table I (main results): preprocessing fitted on the training folds only.
- Table II (ablations): each change is compared with the final model, fold by
  fold. For speed, the preprocessing (which never uses the labels) is fitted
  once on all training rows. Logistic regressions are then
  fitted on the training rows of each fold through a 0/1 row mask, which gives
  the same solution as logistic_newton on the selected rows without copying them.
- Decision thresholds are chosen on the training folds only, never on the
  evaluated fold.

Usage:
    python ablations.py            (about 30 minutes on a laptop CPU)
"""

import os

import numpy as np

from cross_validation import add_intercept, best_threshold, build_k_indices, f1_score
from features import fit_transform_interactions, transform_interactions
from implementations import ridge_regression
from model import logistic_newton, sigmoid
from preprocess import fit_transform, load_column_names, transform
from specifications import SPECS, _cat
from utils import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

K_FOLD = 5
SEED = 1
LAMBDA_RIDGE = 3e-3  # ridge_search.py
LAMBDA_NO_AGE = 1e-3  # model_search.py, logistic regression without age interactions
LAMBDA = 1e-4  # model_search.py, final model (same value as run.py)


# =============================================================================
# Helpers
# =============================================================================


def standardize(a):
    """Standardize the columns of a (constant columns are left at 0)."""
    std = a.std(axis=0)
    std[std == 0] = 1.0
    return (a - a.mean(axis=0)) / std


def newton_masked(tx, y, mask, lambda_, max_iters=30, tol=1e-6, chunk=40000):
    """logistic_newton fitted on the rows where mask == 1, without copying them.

    The Hessian is accumulated by chunks of rows to keep memory low.
    """
    n_tr, d = mask.sum(), tx.shape[1]
    w = np.zeros(d)
    penalty = np.full(d, lambda_)
    penalty[0] = 0.0
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        grad = tx.T @ (mask * (p - y)) / n_tr + penalty * w
        s = mask * p * (1 - p) / n_tr
        hess = np.diag(penalty)
        for i in range(0, len(y), chunk):
            a = tx[i : i + chunk]
            hess += (a.T * s[i : i + chunk]) @ a
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def cv_logistic_masked(tx, y, k_indices, lambda_):
    """F1 per fold of a logistic regression on a fixed design matrix."""
    scores = []
    for val in k_indices:
        mask = np.ones(len(y))
        mask[val] = 0.0
        p = sigmoid(tx @ newton_masked(tx, y, mask, lambda_))
        train = mask == 1
        t = best_threshold(y[train], p[train])
        scores.append(f1_score(y[val], p[val] >= t))
    return np.array(scores)


def report(title, rows, reference=None):
    """Print F1 mean +/- std, and paired difference +/- s.e. to the reference row."""
    print(f"\n{title}")
    ref = rows[reference] if reference is not None else None
    for name, f in rows.items():
        line = f"  {name:<46} F1 {f.mean():.4f} +/- {f.std():.4f}"
        if ref is not None and name != reference:
            d = f - ref
            line += f" | diff {d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f}"
        print(line, flush=True)


# =============================================================================
# Table I: preprocessing fitted on the training folds only
# =============================================================================


def table_main(x, y, names, k_indices):
    """Constant, ridge, logistic, + age interactions (final model)."""
    rows = {
        name: []
        for name in (
            f"Ridge regression (lambda={LAMBDA_RIDGE:.0e})",
            f"Logistic regression (lambda={LAMBDA_NO_AGE:.0e})",
            f"+ age interactions (lambda={LAMBDA:.0e}), final",
        )
    }
    ridge, logistic, final = list(rows)
    for k, val in enumerate(k_indices):
        tr = np.concatenate([idx for i, idx in enumerate(k_indices) if i != k])
        tx_tr, feats, prep = fit_transform(x[tr], names)
        tx_val, _ = transform(x[val], prep)
        fits = []
        a_tr, a_val = add_intercept(tx_tr), add_intercept(tx_val)
        w, _ = ridge_regression(y[tr], a_tr, LAMBDA_RIDGE)
        fits.append((ridge, a_tr @ w, a_val @ w))
        w = logistic_newton(y[tr], a_tr, LAMBDA_NO_AGE)
        fits.append((logistic, sigmoid(a_tr @ w), sigmoid(a_val @ w)))
        del a_tr, a_val
        tx_tr, _, stats = fit_transform_interactions(tx_tr, feats)
        tx_val = transform_interactions(tx_val, feats, stats)
        tx_tr, tx_val = add_intercept(tx_tr), add_intercept(tx_val)
        w = logistic_newton(y[tr], tx_tr, LAMBDA)
        fits.append((final, sigmoid(tx_tr @ w), sigmoid(tx_val @ w)))
        for row, s_tr, s_val in fits:
            t = best_threshold(y[tr], s_tr)
            rows[row].append(f1_score(y[val], s_val >= t))
        del tx_tr, tx_val
        print(f"  Table I: fold {k + 1}/{K_FOLD} done", flush=True)
    report(
        "TABLE I (preprocessing fitted per fold)",
        {n: np.array(v) for n, v in rows.items()},
    )


# =============================================================================
# Table II: ablations, preprocessing fitted once on all training rows
# =============================================================================


def design(
    x, names, specs=SPECS, max_missing=0.95, min_missing=0.005, extra=None, sex=False
):
    """Preprocess, append optional extra columns, add age (and sex) interactions."""
    tx, feats, _ = fit_transform(
        x, names, specs=specs, max_missing=max_missing, min_missing=min_missing
    )
    if extra is not None:
        tx = np.hstack([tx, standardize(extra)])
        feats = feats + [f"extra_{j}" for j in range(extra.shape[1])]
    n_base = tx.shape[1]
    tx, feats, _ = fit_transform_interactions(tx, feats)
    blocks = [np.ones((len(tx), 1)), tx]
    if sex:
        s = next(i for i, f in enumerate(feats[:n_base]) if f.startswith("SEX="))
        others = [j for j in range(n_base) if j != s]
        blocks.append(standardize(tx[:, [s]] * tx[:, others]))
    return np.hstack(blocks)


def module_presence(x, names):
    """Presence indicators of the optional-module questions (HAREHAB1 excluded),
    identical columns merged, columns answered by fewer than 100 rows dropped."""
    start, end = names.index("PDIABTST"), names.index("ADANXEV")
    cols = [c for c in names[start : end + 1] if c != "HAREHAB1"]
    present = ~np.isnan(x[:, [names.index(c) for c in cols]])
    _, first = np.unique(present.T, axis=0, return_index=True)
    keep = [j for j in sorted(first) if 100 <= present[:, j].sum() < len(x)]
    return present[:, keep].astype(float)


def harehab1_presence(x, names):
    """Whether HAREHAB1 was answered. It is only asked after a heart attack, so it
    reveals the label (target leak): measured here, excluded from the final model."""
    return (~np.isnan(x[:, [names.index("HAREHAB1")]])).astype(float)


def table_ablations(x, y, names, k_indices):
    """Effect of each change, compared fold by fold with the final model."""
    genhlth_onehot = {**SPECS, "GENHLTH": _cat(na=(7, 9))}
    configs = {
        "reference: final model": dict(),
        "no missing-value indicator": dict(min_missing=2.0),
        "all indicators, drop variables >65% missing": dict(
            max_missing=0.65, min_missing=1e-9
        ),
        "GENHLTH one-hot instead of ordinal": dict(specs=genhlth_onehot),
        "+ module-presence indicators": dict(extra=module_presence(x, names)),
        "+ sex x feature interactions": dict(sex=True),
        "+ HAREHAB1 presence (target leak, excluded)": dict(
            extra=harehab1_presence(x, names)
        ),
    }
    rows = {}
    for name, kw in configs.items():
        rows[name] = cv_logistic_masked(design(x, names, **kw), y, k_indices, LAMBDA)
        print(f"  Table II: {name} done", flush=True)
    report(f"TABLE II (lambda={LAMBDA:.0e})", rows, "reference: final model")


def main():
    x, _, y_train, _, _ = load_data(DATA_DIR)
    y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}
    names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))
    k_indices = build_k_indices(len(y), K_FOLD, SEED)
    print(f"Constant 'healthy' classifier: accuracy {1 - y.mean():.4f}, F1 0")
    table_main(x, y, names, k_indices)
    table_ablations(x, y, names, k_indices)


if __name__ == "__main__":
    main()
