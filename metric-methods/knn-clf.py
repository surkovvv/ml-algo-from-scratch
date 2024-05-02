import pandas as pd
import numpy as np

from typing import Optional, Union


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform') -> None:
        super().__init__()
        self.k_neig = k
        self.train_size: Optional[int] = None
        self.metric = metric
        self.weight = weight

    @staticmethod
    def _calc_distance(test_vector: np.ndarray, train_data: pd.DataFrame, metric: str = 'euclidean') -> pd.Series:
        if metric == 'euclidean':
            distance = ((train_data - test_vector).pow(2)).sum(axis=1).pow(0.5)
        elif metric == 'chebyshev':
            distance = (train_data - test_vector).abs().max(axis=1)
        elif metric == 'manhattan':
            distance = (train_data - test_vector).abs().sum(axis=1)
        elif metric == 'cosine':
            rows_norm = train_data.pow(2).sum(axis=1).pow(0.5)
            vector_norm = np.sqrt(np.sum(np.square(test_vector)))
            distance = 1 - (train_data * test_vector).sum(axis=1) / rows_norm / vector_norm
        else:
            raise NotImplementedError
        return distance
    
    def _calc_weights_for_knn(self, classes: np.ndarray, distances: Optional[np.ndarray], to_return: str = 'class') -> Union[float, int]:
        if self.weight == 'uniform':
            if to_return == 'class':
                mode = classes.mode()
                if mode.shape[0] == 2:
                    result = 1
                else:
                    result = mode[0]
            else:
                result = classes.sum() / classes.shape[0]
        elif self.weight == 'rank':
            weights = 1 / np.arange(1, classes.shape[0] + 1)
            class_weights = np.zeros(2)
            for class_n in range(class_weights.shape[0]):
                class_weights[class_n] = weights[classes == class_n].sum() / weights.sum()
            if to_return == 'class':
                result = np.argmax(class_weights)
            else:
                result = class_weights[1]
        elif self.weight == 'distance':
            weights = 1 / distances
            class_weights = np.zeros(2)
            for class_n in range(class_weights.shape[0]):
                class_weights[class_n] = weights[classes == class_n].sum() / weights.sum()
            if to_return == 'class':
                result = np.argmax(class_weights)
            else:
                result = class_weights[1]
        else:
            raise NotImplementedError
        return result
    
    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k_neig}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MyKNNClf':
        self.train_size = X.shape
        self.train_data = X
        self.train_target = y
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_test_elemns = X.shape[0]
        predictions = np.ones(n_test_elemns, dtype=int)
        for row in range(n_test_elemns):
            distances = self._calc_distance(X.iloc[row], self.train_data, self.metric)
            leastk_indices = np.argsort(distances)[:self.k_neig]
            classes = self.train_target[leastk_indices]
            sorted_distances = np.sort(distances)[:self.k_neig]
            predictions[row] = self._calc_weights_for_knn(classes, sorted_distances, to_return='class')
        return predictions


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_test_elemns = X.shape[0]
        probas = np.ones(n_test_elemns)
        for row in range(n_test_elemns):
            distances = self._calc_distance(X.iloc[row], self.train_data, self.metric)
            leastk_indices = np.argsort(distances)[:self.k_neig]
            classes = self.train_target[leastk_indices]
            sorted_distances = np.sort(distances)[:self.k_neig]
            probas[row] = self._calc_weights_for_knn(classes, sorted_distances, to_return='proba') 
        return probas
