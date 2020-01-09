import numpy as np
import pandas as pd
from typing import Callable


def permutation_importance_one_feature(model: object,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       feature_number: int,
                                       error_func: Callable,
                                       samples_numb: int=1000) -> np.ndarray:
    # TODO dosctrings
    y_pred = model.predict(X).ravel()
    error = error_func(y.ravel(), y_pred.ravel())

    scores = []
    for sample in range(samples_numb):
        shuff_test = X.copy()
        shuff_test[:, feature_number] = np.random.permutation(shuff_test[:, feature_number])

        score = error_func(y.ravel(), model.predict(shuff_test).ravel())
        scores.append(score)

        if sample % 100 == 0:
            print(sample)

    scores = np.asarray(scores)
    importances = (scores - error)/error

    return importances.ravel()


def permutation_importance(model: object,
                           X: np.ndarray,
                           y: np.ndarray,
                           feature_names: list,
                           error_func: Callable,
                           samples_numb: int=1000) -> pd.DataFrame:
    # TODO dosctrings
    # TODO change pace indicator to %

    importances_df = pd.DataFrame()
    for feature_number in range(X.shape[1]):
        feature_importance = permutation_importance_one_feature(model=model,
                                                                X=X,
                                                                y=y,
                                                                feature_number=feature_number,
                                                                error_func=error_func,
                                                                samples_numb=samples_numb)

        importances_df[feature_names[feature_number]] = feature_importance

    return importances_df