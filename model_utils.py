import numpy as np


def get_prediction(model: object,
                   X: np.ndarray,
                   mode: str = 'reg',
                   predict_proba: bool = True) -> np.ndarray:
    """
    Method for prediction depending on problem type - either regression or classification.
    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        mode (str): if classification - "clf", if regression - "reg" - by default set to "reg"
        predict_proba (bool): if classification mode, parameter for calculating probability as prediction, default True

    Returns:
         (numpy.ndarray) prediction values

    """
    if mode == 'reg':
        return model.predict(X).ravel()
    elif mode == 'clf':
        if predict_proba:
            return model.predict_proba(X)[:, 1].ravel()
        else:
            return model.predict(X).ravel()
    else:
        raise ValueError('Wrong mode. Possible modes: "reg" or "clf"')