import pandas as pd
import numpy as np

from my_lin_reg import MyLineReg
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_regression



if __name__ == '__main__':
    my_lin_reg = MyLineReg(metric='r2')
    # data = load_diabetes(as_frame=True)
    # X, y = data['data'], data['target']
    # print(X.head())


    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    my_lin_reg.fit(X, y, 10)
    print(my_lin_reg.get_best_score())