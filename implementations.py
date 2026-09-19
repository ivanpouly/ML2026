import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        The value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return (e @ e) / (2 * len(y))


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        An array of shape (D,), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N,). The target values.
        tx: shape=(N, D). The design matrix.
        initial_w: shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.

    Returns:
        w: shape=(D,). The model parameters obtained after we finished the total number of iterations of GD.
        loss: a scalar denoting the value of the loss function after we finished the total number of iterations of GD.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    return w, compute_loss(y, tx, w)
