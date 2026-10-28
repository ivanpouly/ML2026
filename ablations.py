"""Reproduce the cross-validation numbers of Tables I and II of the report.

All experiments use the same 5 folds (seed 1) and report the F1 score of the
positive class: mean +/- std over folds, and paired differences between two
models as mean +/- standard error over folds.

- Table I (main results): ridge, logistic regression, + age interactions.
- Table II (ablations): each change is compared with the final model, fold by
  fold.
- In both tables, every preprocessing statistic (medians, levels, means,
  standard deviations, selected columns) is fitted on the training folds only,
  and the decision threshold is chosen on the training folds, never on the
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


def standardize(a_tr, a_val):
    """Standardize columns with the training statistics (constant columns -> 0)."""
    mean, std = a_tr.mean(axis=0), a_tr.std(axis=0)
    std[std == 0] = 1.0
    return (a_tr - mean) / std, (a_val - mean) / std


def fold_f1(y_tr, y_val, tx_tr, tx_val, lambda_):
    """Validation F1 of a logistic regression, threshold chosen on the training rows."""
    w = logistic_newton(y_tr, tx_tr, lambda_)
    t = best_threshold(y_tr, sigmoid(tx_tr @ w))
    return f1_score(y_val, sigmoid(tx_val @ w) >= t)


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
    """Ridge, logistic, + age interactions (final model).

    Returns the per-fold F1 of the final model, the reference of Table II.
    """
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
    rows = {n: np.array(v) for n, v in rows.items()}
    report("TABLE I", rows)
    return rows[final]


# =============================================================================
# Table II: one change at a time, compared with the final model
# =============================================================================


def design_fold(
    x_tr,
    x_val,
    names,
    specs=SPECS,
    max_missing=0.95,
    min_missing=0.005,
    extra=None,
    sex=False,
):
    """Training and validation design matrices of one fold, with the intercept.

    Every statistic is fitted on x_tr. `extra` is an optional function
    (x_tr, x_val, names) -> (columns_tr, columns_val) of additional raw columns,
    appended before the age interactions.
    """
    tx_tr, feats, prep = fit_transform(
        x_tr, names, specs=specs, max_missing=max_missing, min_missing=min_missing
    )
    tx_val, _ = transform(x_val, prep)
    if extra is not None:
        e_tr, e_val = standardize(*extra(x_tr, x_val, names))
        tx_tr, tx_val = np.hstack([tx_tr, e_tr]), np.hstack([tx_val, e_val])
        feats = feats + [f"extra_{j}" for j in range(e_tr.shape[1])]
    n_base = tx_tr.shape[1]
    tx_tr, _, stats = fit_transform_interactions(tx_tr, feats)
    tx_val = transform_interactions(tx_val, feats, stats)
    if sex:
        s = next(i for i, f in enumerate(feats) if f.startswith("SEX="))
        others = [j for j in range(n_base) if j != s]
        p_tr, p_val = standardize(
            tx_tr[:, [s]] * tx_tr[:, others], tx_val[:, [s]] * tx_val[:, others]
        )
        tx_tr, tx_val = np.hstack([tx_tr, p_tr]), np.hstack([tx_val, p_val])
        del p_tr, p_val
    return add_intercept(tx_tr), add_intercept(tx_val)


def module_presence(x_tr, x_val, names):
    """Presence indicators of the optional-module questions (HAREHAB1 excluded).

    On the training rows: identical columns are merged and columns answered by
    fewer than 100 rows are dropped; the same columns are kept for x_val.
    """
    start, end = names.index("PDIABTST"), names.index("ADANXEV")
    idx = [names.index(c) for c in names[start : end + 1] if c != "HAREHAB1"]
    present_tr, present_val = ~np.isnan(x_tr[:, idx]), ~np.isnan(x_val[:, idx])
    _, first = np.unique(present_tr.T, axis=0, return_index=True)
    keep = [j for j in sorted(first) if 100 <= present_tr[:, j].sum() < len(x_tr)]
    print(f"    {len(keep)} module-presence indicators", flush=True)
    return present_tr[:, keep].astype(float), present_val[:, keep].astype(float)


def harehab1_presence(x_tr, x_val, names):
    """Whether HAREHAB1 was answered. It is only asked after a heart attack, so it
    reveals the label (target leak): measured here, excluded from the final model."""
    j = names.index("HAREHAB1")
    return (~np.isnan(x_tr[:, [j]])).astype(float), (~np.isnan(x_val[:, [j]])).astype(
        float
    )


def table_ablations(x, y, names, k_indices, reference):
    """Effect of each change, compared fold by fold with the final model."""
    genhlth_onehot = {**SPECS, "GENHLTH": _cat(na=(7, 9))}
    configs = {
        "no missing-value indicator": dict(min_missing=2.0),
        "all indicators, drop variables >65% missing": dict(
            max_missing=0.65, min_missing=1e-9
        ),
        "GENHLTH one-hot instead of ordinal": dict(specs=genhlth_onehot),
        "+ module-presence indicators": dict(extra=module_presence),
        "+ sex x feature interactions": dict(sex=True),
        "+ HAREHAB1 presence (target leak, excluded)": dict(extra=harehab1_presence),
    }
    rows = {"reference: final model": reference}
    for name, kw in configs.items():
        scores = []
        for k, val in enumerate(k_indices):
            tr = np.concatenate([idx for i, idx in enumerate(k_indices) if i != k])
            tx_tr, tx_val = design_fold(x[tr], x[val], names, **kw)
            scores.append(fold_f1(y[tr], y[val], tx_tr, tx_val, LAMBDA))
            del tx_tr, tx_val
        rows[name] = np.array(scores)
        print(f"  Table II: {name} done", flush=True)
    report(f"TABLE II (lambda={LAMBDA:.0e})", rows, "reference: final model")


def main():
    x, _, y_train, _, _ = load_data(DATA_DIR)
    y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}
    names = load_column_names(os.path.join(DATA_DIR, "x_train.csv"))
    k_indices = build_k_indices(len(y), K_FOLD, SEED)
    print(f"Constant 'healthy' classifier: accuracy {1 - y.mean():.4f}, F1 0")
    reference = table_main(x, y, names, k_indices)
    table_ablations(x, y, names, k_indices, reference)


if __name__ == "__main__":
    main()
