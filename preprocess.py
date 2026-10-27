import numpy as np

from specifications import SPECS


def load_column_names(csv_path):
    """Return the column names from x_train.csv or x_test.csv, without the Id column.
    Args:
        csv_path (str): path to the CSV file
    Returns:
        list of str: column names, in the same order as the CSV file
    """
    with open(csv_path) as f:
        return f.readline().strip().split(",")[1:]


def _special_columns(x, idx):
    """We derive two features from three special features
    Args:
        x: shape=(N, D). The raw data matrix.
        idx: dict {column name: index in x}.
    Returns:
        dict {derived column name: array of shape (N,)}.
    """
    disp = x[:, idx["DISPCODE"]]  # 1100 = completed, 1200 = partial complete
    land = x[:, idx["NUMADULT"]]  # landline: 1-5, 6 or more coded 6-99
    cell = x[:, idx["HHADULT"]].copy()  # cell phone: 1-76, 77 / 99 = DK / refused

    cell[np.isin(cell, (77, 99))] = np.nan

    return {
        "IS_PARTIAL": (disp == 1200).astype(float),
        "HH_ADULTS": np.minimum(np.where(np.isnan(land), cell, land), 6),
    }


def _recode(col, spec):
    """Take a column, recode its value according to its specification, and return a new array.
    Args:
        col: shape=(N,). The column to recode.
        spec: dict. The variable specification (from SPECS).
    Returns:
        shape=(N,). The recoded column.
    """
    col = col.copy()
    if spec["na"]:
        col[np.isin(col, spec["na"])] = np.nan
    if spec["zero"]:
        col[np.isin(col, spec["zero"])] = 0.0
    if spec["valid"]:
        low, high = spec["valid"]
        col[(col < low) | (col > high)] = np.nan
    if spec["log"]:
        col = np.log1p(col)
    return col


def _recoded_columns(x, names, specs):
    """Given a design matrix, and a subset of its column names, return a dict of recoded columns.
    Args:
        x: shape=(N, D). The raw data matrix.
        names: list of D column names.
        specs: dict {column name: variable specification}.
    Returns:
        dict {column name: recoded column of shape (N,)}.
    """
    idx = {name: i for i, name in enumerate(names)}
    special = _special_columns(x, idx)
    cols = {}
    for name, spec in specs.items():
        if name in special:
            cols[name] = _recode(special[name], spec)
        elif name in idx:
            cols[name] = _recode(x[:, idx[name]], spec)
    return cols


def _feature_names(plan):
    """Given a list of features, return the list of features with the additional
    feature names needed, e.g. if the feature is categorical, or if a missing indicator is added.
    Args:
        plan: list of dicts. Each dict describes a feature to keep, with keys:
            name: str. The column name.
            kind: str. "num" or "cat".
            add_missing: bool. Whether to add a missing indicator.
            levels: list of float. The levels of a categorical variable (without the reference).
            fill: float. The value to impute for a numeric variable.
    Returns:
        list of str. The names of the needed features.
    """
    names = []
    for step in plan:
        if step["kind"] == "num":
            names.append(step["name"])
        else:
            names += [f"{step['name']}={v:g}" for v in step["levels"]]
        if step["add_missing"]:
            names.append(f"{step['name']}_missing")
    return names


def _encode(cols, plan, n_rows):
    """Build a design matrix from a list of features, given the recoded columns and
    the plan of which features to add and how to encode them.
    Args:
        cols: dict {column name: recoded column of shape (N,)}.
        plan: list of dicts. Each dict describes a feature to keep, with keys:
            name: str. The column name.
            kind: str. "num" or "cat".
            add_missing: bool. Whether to add a missing indicator.
            levels: list of float. The levels of a categorical variable (without the reference).
            fill: float. The value to impute for a numeric variable.
        n_rows: int. The number of rows in the design matrix.
    Returns:
        X: shape=(N, F). The design matrix.
        names: list of str. The names of the features in the design matrix.
    """
    names = _feature_names(plan)
    X = np.empty((n_rows, len(names)))
    j = 0
    for step in plan:
        col = cols[step["name"]]
        miss = np.isnan(col)
        if step["kind"] == "num":
            X[:, j] = np.where(miss, step["fill"], col)
            j += 1
        else:
            for level in step["levels"]:
                X[:, j] = col == level
                j += 1
        if step["add_missing"]:
            X[:, j] = miss
            j += 1
    return X, names


def fit_transform(x, names, specs=SPECS, max_missing=0.95, min_missing=0.005):
    """Given the raw data, return a design matrix with the features encoded and standardized.
    Args:
        x: shape=(N, D). The raw data matrix.
        names: list of D column names.
        specs: dict {column name: variable specification}.
        max_missing: float. The maximum fraction of missing values allowed for a feature to be kept.
        min_missing: float. The minimum fraction of missing values for a feature to have a missing indicator added.
    Returns:
        X: shape=(N, F). The design matrix with the features encoded and standardized.
        features: list of F feature names.
        prep: dict. The preprocessing information needed to apply the same transformation to new data.
    """
    cols = _recoded_columns(x, names, specs)
    plan = []
    for name, col in cols.items():
        miss = np.isnan(col)
        rate = miss.mean()
        if rate > max_missing:
            continue
        step = {
            "name": name,
            "kind": specs[name]["kind"],
            "add_missing": rate > min_missing,
        }
        if step["kind"] == "num":
            step["fill"] = np.nanmedian(col)
        else:
            levels, counts = np.unique(col[~miss], return_counts=True)
            reference = levels[np.argmax(counts)]
            step["reference"] = reference
            step["levels"] = [v for v in levels if v != reference]
        plan.append(step)

    X, features = _encode(cols, plan, x.shape[0])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    X -= mean
    X /= std

    prep = {
        "names": names,
        "specs": specs,
        "plan": plan,
        "mean": mean,
        "std": std,
        "features": features,
    }
    return X, features, prep


def transform(x, prep):
    """Given the raw data, return a design matrix with the features encoded and standardized,
    using the preprocessing information from a previous fit_transform call.
    Args:
        x: shape=(N, D). The raw data matrix.
        prep: dict. The preprocessing information from a previous fit_transform call.
    Returns:
        X: shape=(N, F). The design matrix with the features encoded and standardized.
        features: list of F feature names.
    """
    cols = _recoded_columns(x, prep["names"], prep["specs"])
    X, features = _encode(cols, prep["plan"], x.shape[0])
    X -= prep["mean"]
    X /= prep["std"]
    return X, features
