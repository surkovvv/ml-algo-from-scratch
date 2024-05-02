import pandas as pd
import numpy as np
import random

from typing import Optional, Union, Callable


class MyLineReg:
    """
    Linear Regression

    Parameters
    ----------
    n_iter : int, optional
        number of iterations of gradient descent, by default 100
    learning_rate : Union[float, Callable], optional
        Learning rate.
        If given lambda-function, learning_rate calculated using this function each step, by default 0.01
    weights : np.ndarray, optional
        Model weights, by default None
    metric : str, optional
        Takes one of these values: mae, mse, rmse, mape, r2, by default None
    reg : str, optional
        Regularization type. Takes one of these values: l1, l2, elasticnet, by default None
    l1_coef : float, optional
        L1 coef of regularization. From 0.0 to 1.0, by default 0
    l2_coef : float, optional
        L2 coef of regularization. From 0.0 to 1.0, by default 0
    sgd_sample : Union[int, float], optional
        Amount of samples used in gradient calculation.
        Float from 0.0 to 1.0 or int, by default None
    random_state : int, optional
        Seed, by default 42
    """
    def __init__(self, 
        weights: Optional[np.ndarray] = None, 
        n_iter: int = 100, 
        learning_rate: Union[float, Callable[[int], int]] = 0.1, 
        metric: Optional[str] = None, 
        reg: Optional[str] = None, 
        l1_coef: float = 0.,
        l2_coef: float = 0.,
        sgd_sample: Optional[Union[int, float]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.lr = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def _calc_metric(self, y_pred: pd.Series, y_true: pd.Series) -> Optional[float]:
        if self.metric == 'mae':
            return np.abs(y_pred - y_true).mean()
        elif self.metric == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_true) ** 2))
        elif self.metric == 'r2':
            y_mean = np.mean(y_true)
            return 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_mean) ** 2)
        elif self.metric == 'mape':
            return 100 * np.mean(np.abs((y_pred - y_true) / y_true))
        elif self.metric is None:
            return None
        else:
            raise NotImplementedError
    
    def _calc_loss(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        loss = (y_pred - y_true).T @ (y_pred - y_true) / y_pred.shape[0]
        if self.reg == 'l1':
            loss +=  self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            loss += self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == 'elasticnet':
            loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == None:
            loss = loss
        else:
            raise NotImplementedError
        return loss

    def _calc_grad(self, X: pd.DataFrame, y_pred: pd.Series, y_true: pd.Series) -> pd.Series:
        grad = 2 * X.T @ (y_pred - y_true) / X.shape[0]
        if self.reg == 'l1':
            grad += self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            grad += 2 * self.l2_coef * self.weights
        elif self.reg == 'elasticnet':
            grad += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        elif self.reg == None:
            grad = grad
        else:
            raise NotImplementedError
        return grad
    
    def _calc_lr(self, n_iter: int) -> float:
        if type(self.lr) == float:
            return self.lr
        elif callable(self.lr) and isinstance(self.lr, type(lambda: None)):
            return self.lr(n_iter)
        else:
            raise NotImplementedError
    
    def _get_sgd_sample(self, n_elem: int) -> int:
        if self.sgd_sample is None:
            return n_elem
        elif type(self.sgd_sample) == int:
            return self.sgd_sample
        elif type(self.sgd_sample) == float:
            return int(self.sgd_sample * n_elem)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.lr}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False) -> 'MyLineReg':
        random.seed(self.random_state)
        n_elem, n_feat = X.shape
        X.insert(0, 'intercept', np.ones(n_elem))
        self.weights = np.ones(n_feat + 1)
        sgd_sample = self._get_sgd_sample(n_elem)

        for iter in range(1, self.n_iter + 1):
            sample_rows_idx = random.sample(range(n_elem), sgd_sample)
            y_pred = X @ self.weights
            loss = self._calc_loss(y_pred, y)
            grad = self._calc_grad(X.iloc[sample_rows_idx], y_pred.iloc[sample_rows_idx], y.iloc[sample_rows_idx])
            lr = self._calc_lr(iter)
            self.weights -= lr * grad
            self.last_metric_value = self._calc_metric(X @ self.weights, y)
            if verbose and iter % verbose == 0:
                print(f'step {iter}, loss = {loss} | {self.metric} = {self.last_metric_value}| lr = {lr}')

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X.insert(0, 'intercept', np.ones(X.shape[0]))
        y_pred = X @ self.weights
        return y_pred.values
    
    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def get_best_score(self) -> float:
        return self.last_metric_value
