import numpy as np
import pandas as pd


def ale_plot(model: object,
             X: np.ndarray,
             feature_number: int,
             resolution: int=100,
             delta_split: int=None,
             feature_type: str='continuous') -> pd.DataFrame:
    """
        Function outputs Accumulated Local Effects Plot for given feature number. It handles continuous and ordered
        categorical variables. Unordered categorical variables not supported.
        Implementation based on https://christophm.github.io/interpretable-ml-book/ale.html

        Args:
            model: (object) fitted model with standard predict(X) public method
            X: (numpy.ndarray) multidimensional array of input to the given model
            feature_number: (int) index of feature to compute PDP, where feature vector will be X[:, feature_number]
            resolution: (int) default 100, number of splits within given feature
            delta_split: (int) default None, number of examples in one interval, if given, resolution is computed based
                        on delta_split
            feature_type: (str) type of feature - either "discrete" or "continuous"
        Returns:
            (pandas.DataFrame) dataframe with two columns: feature intervals and ALE values.
    """
    X = X[np.argsort(X[:, feature_number])]

    if feature_type == 'continuous':
        splitted_batches = __get_splitted_batches_continuous(X, resolution, delta_split)
    elif feature_type == 'discrete':
        splitted_batches = __get_splitted_batches_discrete(X, feature_number)
    else:
        raise ValueError('Wrong feature_type!')

    feature_vals = list()
    local_effect_vals = list()
    for batch_instances in splitted_batches:

        batch = batch_instances.copy()
        feature_min = batch[:, feature_number].min()
        feature_max = batch[:, feature_number].max()

        batch[:, feature_number] = feature_min
        y_pred_min = model.predict(batch)

        batch[:, feature_number] = feature_max
        y_pred_max = model.predict(batch)

        y_diff = y_pred_max - y_pred_min
        local_effect_value = y_diff.mean()

        if feature_type == 'continuous':
            feature_vals.append(feature_max)

        elif feature_type == 'discrete':
            feature_vals.append(feature_min)

        local_effect_vals.append(local_effect_value)

    accumulated_local_effect = np.cumsum(np.array(local_effect_vals), axis=0)
    accumulated_local_effect = accumulated_local_effect - accumulated_local_effect.mean()

    ale_plot_df = pd.DataFrame({'feature': feature_vals,
                                'ALE': accumulated_local_effect})

    return ale_plot_df


def __get_splitted_batches_continuous(X_sorted: np.ndarray,
                                      resolution: int,
                                      delta_split: int=None) -> list:
    """
        Function splits dataset into batch list for ALE plot using frequency strategy for continuous feature.

        Args:
            X_sorted: (numpy.ndarray) multidimensional array of input to the given model sorted by feature of interest
            resolution: (int) default 100, number of splits within given feature
            delta_split: (int) default None, number of examples in one interval, if given, resolution is computed based
                        on delta_split
        Returns:
            (list)  list of splitted batches.
    """
    if delta_split:
        resolution = int(X_sorted.shape[0] / delta_split)
    else:
        delta_split = int(X_sorted.shape[0] / resolution)

    if delta_split < 2:
        print('[WARNING] delta_split too small - used delta_split=2.')
        delta_split = 2
        resolution = int(X_sorted.shape[0] / delta_split)

    temp_index_list = np.array([delta_split] * resolution)
    split_index_list = np.cumsum(temp_index_list, axis=0)

    splitted_batches = np.split(X_sorted, indices_or_sections=split_index_list[:-1])
    return splitted_batches


def __get_splitted_batches_discrete(X_sorted: np.ndarray,
                                    feature_number: int) -> list:
    """
        Function splits dataset into batch list for ALE plot using unique values strategy for ordered discrete feature.

        Args:
            X_sorted: (numpy.ndarray) multidimensional array of input to the given model sorted by feature of interest
            feature_number: (int) index of feature to compute PDP, where feature vector will be X[:, feature_number]
        Returns:
            (list) list of splitted batches.
    """
    unique_vals, unique_indexes = np.unique(X_sorted[:, feature_number], return_index=True)
    splitted_instances = np.split(X_sorted, indices_or_sections=unique_indexes[1:])

    splitted_batches = list()
    for batch_number, batch_instances in enumerate(splitted_instances[:-1]):

        current_batch = batch_instances.copy()
        next_batch = splitted_instances[batch_number + 1].copy()

        batch = np.concatenate([current_batch, next_batch], axis=0)
        splitted_batches.append(batch)

    return splitted_batches
