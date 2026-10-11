"""
Models used in the project.

- SGDLinearRegression: plain linear regression trained with mini-batch SGD
  (a from-scratch numpy re-implementation of the paper's TF LinearRegressor).
- paper_design_matrix: one-hot(prefix) + one-hot(type) + one-hot(prefix x type)
- make_model: the improved models trained on log(CPU time).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PREFIX_VOCAB = ["bazelci/", "examples/", "scripts/", "site/", "src/conditions",
                "src/java_tools", "src/main", "src/test", "src/tools",
                "third_party/", "tools/"]
TYPE_VOCAB = ["JAVA", "C/C++", "Starlark", "python", "HTML/CSS/JS"]


def paper_design_matrix(df, cross=True):
    """One-hot encode the paper's two categorical features (+ their cross).

    A feature cross of two one-hot vectors (11 prefixes x 5 types) is a
    55-dim one-hot vector with a single 1 marking the (prefix, type) pair,
    which lets a LINEAR model give each combination its own weight."""
    cols = []
    for p in PREFIX_VOCAB:
        cols.append((df["prefix"] == p).astype(float).rename(f"prefix={p}"))
    for t in TYPE_VOCAB:
        cols.append((df["type"] == t).astype(float).rename(f"type={t}"))
    if cross:
        for p in PREFIX_VOCAB:
            for t in TYPE_VOCAB:
                cols.append(((df["prefix"] == p) & (df["type"] == t))
                            .astype(float).rename(f"{p} x {t}"))
    return pd.concat(cols, axis=1)


class SGDLinearRegression:
    """y_hat = w.x + b, fitted by minimising mean squared error with
    mini-batch stochastic gradient descent.

    Gradient of MSE for a batch B:
        dL/dw = (2/|B|) * X_B^T (X_B w + b - y_B)
        dL/db = (2/|B|) * sum(X_B w + b - y_B)
    """

    def __init__(self, lr=0.01, batch_size=16, epochs_per_period=10, periods=10, seed=0):
        self.lr, self.batch_size = lr, batch_size
        self.epochs_per_period, self.periods = epochs_per_period, periods
        self.rng = np.random.default_rng(seed)
        self.history = []  # (train_rmse, val_rmse) after each period

    def predict(self, X):
        return X @ self.w + self.b

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        n = len(y)
        for _ in range(self.periods):
            for _ in range(self.epochs_per_period):
                order = self.rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = order[start:start + self.batch_size]
                    err = self.predict(X[idx]) - y[idx]
                    self.w -= self.lr * 2 * X[idx].T @ err / len(idx)
                    self.b -= self.lr * 2 * err.mean()
            tr = rmse(y, self.predict(X))
            va = rmse(y_val, self.predict(np.asarray(X_val, float))) if X_val is not None else None
            self.history.append((tr, va))
        return self


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def make_model(name, seed=0):
    """Improved models. All are trained on log(CPU ms) because build times are
    heavily right-skewed (median 1.8 s, max 45 s); predictions are exp()'d back."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                     random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                         subsample=0.8, loss="huber", random_state=seed)
    raise ValueError(name)
