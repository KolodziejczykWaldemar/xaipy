import itertools

import numpy as np
import pandas as pd


def plot_pdp(model: object,
             X: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type: str = 'continuous',
             resolution: int = 100,
             samples_number: int = -1
             ) -> pd.DataFrame:
    """
    Function outputs Partial Dependence Plot for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/pdp.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_number: (int) index of feature to compute PDP, where feature vector will be X[:, feature_number]
        feature_name: (str) name of feature
        feature_type: (str) type of feature - either "discrete" or "continuous"
        resolution: (int) resolution of ICE plot, default 100
        samples_number: (int) number of samples to draw (without replacement) from dataset, default set to -1 (take all)
    Returns:
        (pandas.DataFrame) dataframe with one column of PDP values.
    """

    if samples_number > 0:
        sample_indexes = np.random.randint(X.shape[0], size=samples_number)
        X = X[sample_indexes, :]

    sampled_values = __sample_space(x=X[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=resolution)
    sample_resolution = sampled_values.shape[0]

    stacked_instances = np.empty((0, X.shape[1]), float)
    other_features = np.delete(X,
                               feature_number,
                               axis=1)
    for i, row in enumerate(other_features):
        copied_instances = np.repeat(row.reshape(1, -1),
                                     sample_resolution,
                                     axis=0)
        concatenated_instances = np.insert(copied_instances,
                                           feature_number,
                                           sampled_values.ravel(),
                                           axis=1)
        stacked_instances = np.append(stacked_instances,
                                      concatenated_instances,
                                      axis=0)

    y_pred = model.predict(stacked_instances).ravel()
    feature_results = pd.DataFrame({feature_name: stacked_instances[:, feature_number],
                                    'output': y_pred})
    mean_outputs = feature_results.groupby([feature_name]).mean()

    return mean_outputs


def plot_pdp_2d(model: object,
                X: np.ndarray,
                feature_numbers: list,
                feature_names: list,
                feature_types: list = 'continuous',
                resolution: int = 100,
                samples_number: int = -1) -> pd.DataFrame:
    """
    Function outputs Partial Dependence Plot for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/pdp.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_numbers: (list) list of feature indexes to compute PDP for, where feature vector will be X[:, feature_numbers]
        feature_names: (list) list of feature names
        feature_types: (str) list of feature types - either "discrete" or "continuous"
        resolution: (int) resolution of ICE plot, default 100
        samples_number: (int) number of samples to draw (without replacement) from dataset, default set to -1 (take all)
    Returns:
        (pandas.DataFrame) dataframe with indexes as first feature, columns as second feature and mean PDP values.
    """

    if samples_number > 0:
        sample_indexes = np.random.randint(X.shape[0], size=samples_number)
        X = X[sample_indexes, :]

    sampled_space = __sample_2d_space(X[:, feature_numbers],
                                      feature_types,
                                      sample_resolution=resolution)
    sample_resolution = sampled_space.shape[0]

    stacked_instances = np.empty((0, X.shape[1]), float)

    for i, row in enumerate(X):
        copied_instances = np.repeat(row.reshape(1, -1),
                                     sample_resolution,
                                     axis=0)
        copied_instances[:, feature_numbers[0]] = sampled_space[:, 0]
        copied_instances[:, feature_numbers[1]] = sampled_space[:, 1]

        stacked_instances = np.append(stacked_instances,
                                      copied_instances,
                                      axis=0)

    y_pred = model.predict(stacked_instances).ravel()
    feature_results = pd.DataFrame({feature_names[0]: stacked_instances[:, feature_numbers[0]],
                                    feature_names[1]: stacked_instances[:, feature_numbers[1]],
                                    'output': y_pred})

    pdp_map = pd.pivot_table(feature_results,
                             values='output',
                             index=[feature_names[0]],
                             columns=[feature_names[1]],
                             aggfunc=np.mean)

    return pdp_map


def plot_ice(model: object,
             X: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type: str = 'continuous',
             resolution: int = 100,
             samples_number: int = -1) -> pd.DataFrame:
    """
    Function outputs Individual Conditional Expectation for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/ice.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_number: (int) index of feature to compute ICE, where feature vector will be X[:, feature_number]
        feature_name: (str) name of feature
        feature_type: (str) type of feature - either "discrete" or "continuous"
        resolution: (int) resolution of ICE plot, default 100
        samples_number: (int) number of samples to draw (without replacement) from dataset, default set to -1 (take all)
    Returns:
        (pandas.DataFrame) dataframe with ICE values.
    """

    if samples_number > 0:
        sample_indexes = np.random.randint(X.shape[0], size=samples_number)
        X = X[sample_indexes, :]

    sampled_values = __sample_space(x=X[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=resolution)
    sample_resolution = sampled_values.shape[0]

    stacked_instances = np.empty((0, X.shape[1]), float)
    other_features = np.delete(X,
                               feature_number,
                               axis=1)

    row_indicator = []
    for i, row in enumerate(other_features):
        copied_instances = np.repeat(row.reshape(1, -1),
                                     sample_resolution,
                                     axis=0)
        concatenated_instances = np.insert(copied_instances,
                                           feature_number,
                                           sampled_values.ravel(),
                                           axis=1)
        stacked_instances = np.append(stacked_instances,
                                      concatenated_instances,
                                      axis=0)

        row_indicator += ((np.ones(sample_resolution) * i).ravel().tolist())

    y_pred = model.predict(stacked_instances).ravel()
    feature_results = pd.DataFrame({feature_name: stacked_instances[:, feature_number],
                                    'output': y_pred,
                                    'sample_id': row_indicator})

    samples_groups_df = pd.DataFrame(
        {feature_name: feature_results[feature_results['sample_id'] == 0][feature_name].values})
    samples_groups_df.set_index(feature_name, drop=True, inplace=True)

    for slice_num in range(X.shape[0]):
        samples_groups_df[str(slice_num)] = feature_results[feature_results['sample_id'] == slice_num]['output'].values

    return samples_groups_df


def __sample_space(x: np.ndarray,
                   feature_type: str,
                   sample_resolution: int = 100) -> np.ndarray:
    """
    Function samples space according to feature_type.

    Args:
        x: (numpy.ndarray) 1-D array of one feature values
        feature_type: (str) type of feature - either "discrete" or "continuous"
        sample_resolution: (int) size of sampled space
    Returns:
        (numpy.ndarray) vector of sample space.
    """
    if feature_type == 'continuous':
        min_value = x.min()
        max_value = x.max()
        sampled_values = np.linspace(start=min_value, stop=max_value, num=sample_resolution)
    elif feature_type == 'discrete':
        sampled_values = np.unique(x, return_index=False)

    return sampled_values


def __sample_2d_space(features: np.ndarray,
                      feature_types: list,
                      sample_resolution: int=100) -> np.ndarray:
    """
    Function samples 2D space according to feature_types.

    Args:
        features: (numpy.ndarray) 2-D array of two feature values features.shape[1] = 2
        feature_types: (list) list of feature types - either "discrete" or "continuous"
        sample_resolution: (int) size of sampled space
    Returns:
        (numpy.ndarray) vector of sample space with shape of (:, 2).
    """
    x1 = features[:, 0]
    x2 = features[:, 1]

    sampled_x1 = __sample_space(x1, feature_types[0], sample_resolution=sample_resolution)
    sampled_x2 = __sample_space(x2, feature_types[1], sample_resolution=sample_resolution)

    combinations_list = list(itertools.product(sampled_x1, sampled_x2))
    sample_space = np.asarray(combinations_list)
    return sample_space

