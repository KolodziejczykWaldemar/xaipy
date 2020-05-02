import sys

import numpy as np
import pandas as pd
import progressbar as pb

from typing import Callable

from model_utils import get_prediction


def permutation_importance_one_feature(model: object,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       feature_number: int,
                                       error_func: Callable,
                                       mode: str = 'reg',
                                       samples_numb: int = 1000) -> np.ndarray:
    """
    Function outputs permutation importance for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/feature-importance.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        y: (numpy.ndarray) array of outputs matched to X matrix
        feature_number: (int) index of feature to compute permutation importance, where feature vector will be X[:, feature_number]
        error_func: (function) pointer to error function, based on which error will be computed
        mode: (str) if classification - "clf", if regression - "reg" - by default set to "reg"
        samples_numb: (int) number of permutation trials
    Returns:
        (numpy.ndarray) vector of increased errors of samples_numb length (0 - 0%, 1=100%).
    """
    y_pred = get_prediction(model, X, mode, predict_proba=False)
    error = error_func(y.ravel(), y_pred.ravel())

    bar = pb.ProgressBar(maxval=samples_numb,
                         widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar.start()

    scores = []
    for sample in range(samples_numb):
        X_shuff = X.copy()
        X_shuff[:, feature_number] = np.random.permutation(X_shuff[:, feature_number])

        score = error_func(y.ravel(), get_prediction(model, X_shuff, mode, predict_proba=False))
        scores.append(score)

        bar.update(sample+1)

    bar.finish()
    del bar
    scores = np.asarray(scores)
    importances = (scores - error)/error

    return importances.ravel()


def permutation_importance(model: object,
                           X: np.ndarray,
                           y: np.ndarray,
                           feature_names: list,
                           error_func: Callable,
                           mode: str = 'reg',
                           samples_numb: int=1000) -> pd.DataFrame:
    """
    Function outputs permutation importance for all features within X matrix. Implementation based on
    https://christophm.github.io/interpretable-ml-book/feature-importance.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        y: (numpy.ndarray) array of outputs matched to X matrix
        feature_names: (list) list of feature names
        error_func: (function) pointer to error function, based on which error will be computed
        mode: (str) if classification - "clf", if regression - "reg" - by default set to "reg"
        samples_numb: (int) number of permutation trials
    Returns:
        (pandas.DataFrame) dataframe with columns named as feature_names containing vector of increased errors of
        samples_numb length (0 - 0%, 1=100%).
    """

    importances_df = pd.DataFrame()
    for feature_number in range(X.shape[1]):

        feature_importance = permutation_importance_one_feature(model=model,
                                                                X=X,
                                                                y=y,
                                                                feature_number=feature_number,
                                                                error_func=error_func,
                                                                mode=mode,
                                                                samples_numb=samples_numb)

        importances_df[feature_names[feature_number]] = feature_importance
    return importances_df
