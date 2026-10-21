import os
import numpy as np

from helpers import load_csv_data


def load_data(data_dir):
    """The first time, load the dataset from CSV files and save it in binary.
    Later, load the dataset from binary. The resulting arrays are saved
    to data_dir/cache.npz.

    Args:
        data_dir: string. Folder containing x_train.csv, y_train.csv and x_test.csv.

    Returns:
        x_train, x_test, y_train, train_ids, test_ids: np.array. The same arrays
        returned by load_csv_data.
    """
    cache = os.path.join(data_dir, "cache.npz")
    if os.path.exists(cache):
        with np.load(cache) as d:
            return (
                d["x_train"],
                d["x_test"],
                d["y_train"],
                d["train_ids"],
                d["test_ids"],
            )
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_dir)
    np.savez(
        cache,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return x_train, x_test, y_train, train_ids, test_ids
