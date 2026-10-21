import numpy as np


def compute_mse(y, tx, w):
    """Compute Mean Square Error.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        loss: scalar. The loss corresponding to the input parameters w.
    """
    e = y - tx @ w
    N = len(y)
    return (e @ e) / (2 * N)


def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w, with respect to the MSE.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        gradient: shape (D,). The gradient of the loss at w.
    """
    e = tx @ w - y
    N = len(y)
    return tx.T @ e / N


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm, with respect to the MSE.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: scalar. The number of steps to run.
        gamma: scalar. The stepsize.

    Returns:
        w: shape=(D,). The weight vector after max_iters iterations of the algorithm.
        loss: scalar. The loss value at the last weight vector.
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient_mse(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm, with respect to the MSE.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: scalar. The number of steps to run.
        gamma: scalar. The stepsize.

    Returns:
        w: shape=(D,). The weight vector after max_iters iterations of the algorithm.
        loss: scalar. The loss value at the last weight vector.
    """
    rng = np.random.default_rng(0)
    N = len(y)
    w = initial_w
    for _ in range(max_iters):
        n = rng.integers(N)
        tx_n = tx[n : n + 1]
        y_n = y[n : n + 1]
        w = w - gamma * compute_gradient_mse(y_n, tx_n, w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.

    Returns:
        w: shape=(D,). The weight vector that minimizes the MSE.
        loss: scalar. The loss value at w.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        lambda_: scalar. The regularization parameter.

    Returns:
        w: shape=(D,). The weight vector that minimizes the MSE.
        loss: scalar. The loss value at w.
    """
    N = len(y)
    D = tx.shape[1]
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_error(y, tx, w):
    """The loss function of the logistic regression.

    Args:
        y: shape=(N,). The target values. Values in {0, 1}.
        tx: shape=(N, D). The design matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        loss: scalar. The loss corresponding to the input parameters w.
    """
    pred = tx @ w
    return np.mean(-y * pred + np.logaddexp(0, pred))


def sigmoid(x):
    """Computes the sigmoid of a scalar.

    Args:
        x: scalar.

    Returns:
        y: scalar. The value of the sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))


def logistic_gradient(y, tx, w):
    """Computes the gradient at w, with respect to the logistic regression. y = 0, 1.

    Args:
        y: shape=(N,). The target values. Values in {0, 1}.
        tx: shape=(N, D). The design matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        gradient: shape (D,). The gradient of the loss at w.
    """
    e = sigmoid(tx @ w) - y
    N = len(y)
    return tx.T @ e / N


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The GD algorithm, with respect to the logistic regression. y = 0, 1.

    Args:
        y: shape=(N,). The target values. Values in {0, 1}.
        tx: shape=(N, D). The design matrix.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: scalar. The number of steps to run.
        gamma: scalar. The stepsize.

    Returns:
        w: shape=(D,). The weight vector after max_iters iterations of the algorithm.
        loss: scalar. The loss value at the last weight vector.
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * logistic_gradient(y, tx, w)
    return w, logistic_error(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The GD algorithm, with respect to the regularized logistic regression. y = 0, 1.

    Args:
        y: shape=(N,). The target values. Values in {0, 1}.
        tx: shape=(N, D). The design matrix.
        lambda_: scalar. The regularization parameter.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: scalar. The number of steps to run.
        gamma: scalar. The stepsize.

    Returns:
        w: shape=(D,). The weight vector after max_iters iterations of the algorithm.
        loss: scalar. The loss value at the last weight vector.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, logistic_error(y, tx, w)
