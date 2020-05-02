import math
import itertools
import multiprocessing as mlp

from typing import List
from functools import partial
from collections import namedtuple

import numpy as np

from model_utils import get_prediction


Explanation = namedtuple('Explanation', ['average_prediction', 'actual_prediction', 'shapley_values'])


def get_shapley_values(model: object,
                       X: np.ndarray,
                       x_explain: np.ndarray,
                       mode: str,
                       n_jobs: int = -1) -> Explanation:
    """
        Function outputs Shapley values for given sample that needs to be explained. This method has exponential
        computation complexity in terms of features in dataset. Therefore should not be used with datasets containing
        many variables.
        Implementation based on https://christophm.github.io/interpretable-ml-book/shapley.html

        Args:
            model: (object) fitted model with standard predict(X) public method
            X: (numpy.ndarray) multidimensional array of input to the given model
            x_explain: (numpy.ndarray) one sample of shape (1, features_number) that will be explained
            mode: (str) if classification - "clf", if regression - "reg" - by default set to "reg"
            n_jobs: (int) The number of jobs to run in parallel, by default set to -1, i.e. using all processors.
        Returns:
            (Explanation) named tuple with fields average_prediction, actual_prediction and shapley_values.
    """

    available_cores = mlp.cpu_count()
    if (n_jobs == -1) | (n_jobs > available_cores):
        cores = min(available_cores, X.shape[1])
    else:
        cores = n_jobs

    partial_shap = partial(get_shapley_one_feature, model, X, x_explain, mode)
    pool = mlp.Pool(cores)
    shapley_values = pool.map(partial_shap, list(range(X.shape[1])))

    y_pred = get_prediction(model, x_explain, mode, predict_proba=True)[0]
    y_pred_avg = __get_average_prediction(model, X, [], mode)
    return Explanation(y_pred_avg, y_pred, np.array(shapley_values))


def get_shapley_one_feature(model: object,
                            X: np.ndarray,
                            x_explain: np.ndarray,
                            mode: str,
                            feature_number: int) -> float:
    """
        Function outputs Shapley value for chosen feature for given sample that needs to be explained. This method has
        exponential computation complexity in terms of features in dataset. Therefore should not be used with datasets
        containing many variables.
        Implementation based on https://christophm.github.io/interpretable-ml-book/shapley.html

        Args:
            model: (object) fitted model with standard predict(X) public method
            X: (numpy.ndarray) multidimensional array of input to the given model
            x_explain: (numpy.ndarray) one sample of shape (1, features_number) that will be explained
            mode: (str) if classification - "clf", if regression - "reg" - by default set to "reg"
            feature_number: (int) index of feature that Shapley value with be calculated for
        Returns:
            (float) Shapley value for chosen feature
    """
    features_amount = X.shape[1]
    features = set(range(features_amount))

    shapley_value = 0
    average_prediction = __get_average_prediction(model, X, [], mode)
    for subset_size in range(0, features_amount):
        for subset in itertools.combinations(features.difference(set([feature_number])), subset_size):
            coef = math.factorial(len(subset)) * math.factorial(features_amount - len(subset) - 1) / math.factorial(
                features_amount)
            subset = list(subset)
            coalition_columns = list(zip(subset, x_explain[0, subset].tolist()))
            value_without = __get_average_prediction(model, X, coalition_columns, mode) - average_prediction
            coalition_columns.append((feature_number, x_explain[0, feature_number]))
            value_with = __get_average_prediction(model, X, coalition_columns, mode) - average_prediction
            shapley_value += coef * (value_with - value_without)
    return shapley_value


def __get_average_prediction(model: object,
                             X: np.ndarray,
                             coalition_columns: List[tuple],
                             mode: str) -> float:
    """
        Private function outputs average prediction of model based on dataset with modified feature values according to
        given coalitions of variables.

        Args:
            model: (object) fitted model with standard predict(X) public method
            X: (numpy.ndarray) multidimensional array of input to the given model
            coalition_columns: (list) list of tuples (feature_number, feature_value) for estimating model's average
                                prediction
            mode: (str) if classification - "clf", if regression - "reg" - by default set to "reg"
        Returns:average_prediction
            (float) average prediction value
    """
    X_coal = X.copy()
    for column, value in coalition_columns:
        X_coal[:, column] = value

    y_pred = get_prediction(model, X_coal, mode, predict_proba=True)

    return y_pred.mean(axis=0)


if __name__ == "__main__":
    import time
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor

    dataset = pd.DataFrame(data=load_boston()['data'], columns=load_boston()['feature_names'])
    X = dataset.values5
    y = load_boston()['target']

    model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10)
    model.fit(X, y)

    start = time.time()
    shap = get_shapley_values(model, X, x_explain=X[41:42, :], mode='reg', n_jobs=1)
    print(shap, time.time() - start)
    # part_shap = partial(func, 44)
    # pool = mlp.Pool(5)
    # imp = [1, 2, 3, 4, 5]
    # total_successes = pool.map(part_shap, imp)
    # print(total_successes)
