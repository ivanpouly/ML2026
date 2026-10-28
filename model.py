import numpy as np


def sigmoid(t):
    """Logistic function"""
    return 1 / (1 + np.exp(-t))


def logistic_newton(y, tx, lambda_, max_iters=30, tol=1e-6):
    """Minimize  mean log-loss + lambda_ / 2 * ||w[1:]||^2  with Newton's method.

    The intercept w[0] is not penalized. Each step solves H step = gradient,
    so it converges in a handful of iterations without a learning rate.

    Args:
        y: shape=(N,). Labels in {0, 1}.
        tx: shape=(N, D). Features, first column = intercept (ones).
        lambda_: float > 0. L2 regularization strength.
        max_iters: int. Maximum number of Newton steps.
        tol: float. Stop when the largest weight update is below tol.

    Returns:
        w: shape=(D,). Fitted weights.
    """
    n, d = tx.shape
    w = np.zeros(d)
    penalty = np.full(d, lambda_)
    penalty[0] = 0.0
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        grad = tx.T @ (p - y) / n + penalty * w
        hess = (tx.T * (p * (1 - p))) @ tx / n + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w
