import numpy as np
import pandas as pd


def plot_pdp(model: object,
             features: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type='continuous') -> pd.DataFrame:
    # TODO add docstring

    if features.shape[0] > 1000:
        sample_indexes = np.random.randint(features.shape[0], size=1000)
        features = features[sample_indexes, :]

    sampled_values = __sample_space(feature_vals=features[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=100)
    sample_resolution = sampled_values.shape[0]

    stacked_instances = np.empty((0, features.shape[1]), float)
    other_features = np.delete(features,
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
             features: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type='continuous') -> pd.DataFrame:
    # TODO add docstring

    if features.shape[0] > 1000:
        sample_indexes = np.random.randint(features.shape[0], size=1000)
        features = features[sample_indexes, :]

    sampled_values = __sample_space(feature_vals=features[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=100)
    sample_resolution = sampled_values.shape[0]

    stacked_instances = np.empty((0, features.shape[1]), float)
    other_features = np.delete(features,
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
    for slice_num in range(features.shape[0]):
        samples_groups_df[str(slice_num)] = feature_results[feature_results['sample_id'] == slice_num]['output'].values

    return samples_groups_df


def __sample_space(feature_vals: np.ndarray,
                   feature_type: str,
                   sample_resolution=100) -> np.ndarray:
    if feature_type == 'continuous':
        min_value = feature_vals.min()
        max_value = feature_vals.max()
        sampled_values = np.linspace(start=min_value, stop=max_value, num=sample_resolution)
    elif feature_type == 'discrete':
        sampled_values = np.unique(feature_vals, return_index=False)

    return sampled_values

