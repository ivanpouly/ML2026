import os

from utils import load_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

x_train, x_test, y_train, train_ids, test_ids = load_data(DATA_DIR)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("train_ids shape:", train_ids.shape)
print("test_ids shape:", test_ids.shape)
