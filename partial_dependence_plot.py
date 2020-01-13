import numpy as np
import pandas as pd


def plot_pdp(model: object,
             X: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type='continuous') -> pd.DataFrame:
    """
    Function outputs Partial Dependence Plot for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/pdp.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_number: (int) index of feature to compute PDP, where feature vector will be X[:, feature_number]
        feature_name: (str) name of feature
        feature_type: (str) type of feature - either "discrete" or "continuous"
    Returns:
        # TODO (numpy.ndarray) vector of increased errors of samples_numb length (0 - 0%, 1=100%).
    """

    if X.shape[0] > 1000:
        sample_indexes = np.random.randint(X.shape[0], size=1000)
        X = X[sample_indexes, :]

    sampled_values = __sample_space(feature_vals=X[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=100)
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


def plot_ice(model: object,
             X: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type='continuous') -> pd.DataFrame:
    """
    Function outputs Individual Conditional Expectation for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/ice.html

    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_number: (int) index of feature to compute ICE, where feature vector will be X[:, feature_number]
        feature_name: (str) name of feature
        feature_type: (str) type of feature - either "discrete" or "continuous"
    Returns:
        # TODO (numpy.ndarray) vector of increased errors of samples_numb length (0 - 0%, 1=100%).
    """

    if X.shape[0] > 1000:
        sample_indexes = np.random.randint(X.shape[0], size=1000)
        X = X[sample_indexes, :]

    sampled_values = __sample_space(feature_vals=X[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=100)
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
    samples_groups_df.index = samples_groups_df[feature_name]
    samples_groups_df.drop(feature_name,
                           axis=1,
                           inplace=True)
    for slice_num in range(X.shape[0]):
        samples_groups_df[str(slice_num)] = feature_results[feature_results['sample_id'] == slice_num]['output'].values

    return samples_groups_df


def __sample_space(x: np.ndarray,
                   feature_type: str,
                   sample_resolution=100) -> np.ndarray:
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

