import numpy as np
import pandas as pd
import progressbar2 as pb

from typing import Callable


def permutation_importance_one_feature(model: object,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       feature_number: int,
                                       error_func: Callable,
                                       samples_numb: int=1000) -> np.ndarray:
    """
    Function outputs permutation importance for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/feature-importance.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        y: (numpy.ndarray) array of outputs matched to X matrix
        feature_number: (int) index of feature to compute permutation importance, where feature vector will be X[:, feature_number]
        error_func: (function) pointer to error function, based on which error will be computed
        samples_numb: (int) number of permutation trials
    Returns:
        (numpy.ndarray) vector of increased errors of samples_numb length (0 - 0%, 1=100%).
    """
    y_pred = model.predict(X).ravel()
    error = error_func(y.ravel(), y_pred.ravel())

    bar = pb.ProgressBar(maxval=samples_numb,
                         widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar.start()

    scores = []
    for sample in range(samples_numb):
        shuff_test = X.copy()
        shuff_test[:, feature_number] = np.random.permutation(shuff_test[:, feature_number])

        score = error_func(y.ravel(), model.predict(shuff_test).ravel())
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
        samples_numb: (int) number of permutation trials
    Returns:
        (pandas.DataFrame) dataframe with columns named as feature_names contaiining vector of increased errors of
        samples_numb length (0 - 0%, 1=100%).
    """

    importances_df = pd.DataFrame()
    for feature_number in range(X.shape[1]):
        # print('Processing feature {}'.format(feature_number))
        feature_importance = permutation_importance_one_feature(model=model,
                                                                X=X,
                                                                y=y,
                                                                feature_number=feature_number,
                                                                error_func=error_func,
                                                                samples_numb=samples_numb)

        importances_df[feature_names[feature_number]] = feature_importance

    return importances_df
