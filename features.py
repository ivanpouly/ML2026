"""Age interactions: products of age with every other feature (numpy only).

Risk factors do not weigh the same at 30 and at 70. A linear model cannot
express that on its own; the products age x feature let each feature's
weight change with age.
"""

import numpy as np

AGE = "_AGE80"


def _age_products(tx, features):
    """Products of the age column with every other column of tx."""
    a = features.index(AGE)
    others = [j for j in range(tx.shape[1]) if j != a]
    return tx[:, [a]] * tx[:, others], [f"{AGE}*{features[j]}" for j in others]


def fit_transform_interactions(tx, features):
    """Append the age products to tx, standardized with the statistics of tx.

    Args:
        tx: shape=(N, F). Preprocessed (standardized) training features.
        features: list of F feature names.

    Returns:
        tx: shape=(N, 2F - 1). Features followed by the age products.
        features: list of 2F - 1 names.
        stats: dict with the mean and std of the products, for transform_interactions.
    """
    extra, names = _age_products(tx, features)
    mean, std = extra.mean(axis=0), extra.std(axis=0)
    std[std == 0] = 1.0
    extra -= mean
    extra /= std
    return np.hstack([tx, extra]), features + names, {"mean": mean, "std": std}


def transform_interactions(tx, features, stats):
    """Append the age products to new data, standardized with the TRAINING stats."""
    extra, _ = _age_products(tx, features)
    extra -= stats["mean"]
    extra /= stats["std"]
    return np.hstack([tx, extra])
