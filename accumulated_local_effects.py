import numpy as np
import pandas as pd


def ale_plot(model: object,
             X: np.ndarray,
             feature_number: int,
             resolution: int=100,
             delta_split: int=None) -> pd.DataFrame:
    """
        Function outputs Accumulated Local Effects Plot for given feature number. Implementation based on
        https://christophm.github.io/interpretable-ml-book/ale.html

        Args:
            model: (object) fitted model with standard predict(X) public method
            X: (numpy.ndarray) multidimensional array of input to the given model
            feature_number: (int) index of feature to compute PDP, where feature vector will be X[:, feature_number]
            resolution: (int) default 100, number of splits within given feature
            delta_split: (int) default None, number of examples in one interval, if given, resolution is computed based
                        on delta_split
        Returns:
            (pandas.DataFrame) dataframe with two columns: feature intervals and ALE values.
        """
    X = X[np.argsort(X[:, feature_number])]
    # TODO X.unique().shape[0] < 50: set proper indexes

    if delta_split:
        resolution = int(X.shape[0] / delta_split)
    else:
        delta_split = int(X.shape[0] / resolution)

    if delta_split < 2:
        # TODO throw warning
        delta_split = 2
        resolution = int(X.shape[0] / delta_split)

    temp_index_list = np.array([delta_split] * resolution)
    split_index_list = np.cumsum(temp_index_list, axis=0)

    splitted_instances = np.split(X, indices_or_sections=split_index_list[:-1])

    feature_vals = list()
    local_effect_vals = list()
    for batch_instances in splitted_instances:

        batch = batch_instances.copy()
        feature_min = batch[:, feature_number].min()
        feature_max = batch[:, feature_number].max()

        batch[:, feature_number] = feature_min
        y_pred_min = model.predict(batch)

        batch[:, feature_number] = feature_max
        y_pred_max = model.predict(batch)

        y_diff = y_pred_max - y_pred_min
        local_effect_value = y_diff.mean()

        feature_vals.append(feature_max)
        local_effect_vals.append(local_effect_value)

    accumulated_local_effect = np.cumsum(np.array(local_effect_vals), axis=0)
    accumulated_local_effect = accumulated_local_effect - accumulated_local_effect.mean()

    ale_plot_df = pd.DataFrame({'feature': feature_vals,
                                'ALE': accumulated_local_effect})

    return ale_plot_df