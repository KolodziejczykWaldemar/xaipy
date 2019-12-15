import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

from partial_dependence_plot import plot_ice, plot_pdp

# dataset = pd.DataFrame(data=fetch_california_housing()['data'], columns=fetch_california_housing()['feature_names'])
# features = dataset.values[:10000, :]
# y = fetch_california_housing()['target'][:10000]

from sklearn.datasets import load_boston
print(load_boston()['DESCR'])
dataset = pd.DataFrame(data=load_boston()['data'], columns=load_boston()['feature_names'])
features = dataset.values
y = load_boston()['target']

model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10)
model.fit(features, y)


for f in range(len(dataset.columns)):
    if dataset.columns[f] == 'RAD':
        feature_type = 'discrete'
    else:
        feature_type = 'continuous'

    plot_df = plot_pdp(features=features, model=model, feature_number=f,
                                        feature_name=dataset.columns[f], feature_type=feature_type)

    if feature_type == 'discrete':
        plot_df.plot(figsize=(16, 9), legend=False, marker='o', linewidth=0.2, markersize=3, color='black')
    elif feature_type == 'continuous':
        plot_df.plot(figsize=(16, 9), legend=False, linewidth=1, color='black')
        ax = plt.gca()
        bottom, top = ax.get_ylim()
        indicator_height = 0.05 * (top - bottom)
        ax.plot(dataset[dataset.columns[f]].values, [bottom - indicator_height]*dataset.shape[0], '|', color='k', alpha=0.08)
    plt.show()


for f in range(len(dataset.columns)):
    if dataset.columns[f] == 'RAD':
        feature_type = 'discrete'
    else:
        feature_type = 'continuous'

    res_df = plot_ice(features=features, model=model, feature_number=f, feature_name=dataset.columns[f], feature_type=feature_type)
    if feature_type == 'discrete':
        ax = res_df.plot(figsize=(16, 9), legend=False, marker='o', linewidth=0.5, markersize=1, color='black', alpha=0.08)
        res_df.mean(axis=1).plot(c='red', linewidth=1, legend=False, ax=ax, marker='o', markersize=7)
    elif feature_type == 'continuous':
        ax = res_df.plot(figsize=(16, 9), c='k', alpha=0.08, legend=False, linewidth=0.5)
        res_df.mean(axis=1).plot(c='red', linewidth=5, legend=False, ax=ax)

        bottom, top = ax.get_ylim()
        indicator_height = 0.05 * (top - bottom)
        ax.plot(dataset[dataset.columns[f]].values, [bottom - indicator_height] * dataset.shape[0], '|', color='k',
                alpha=0.08)
    plt.show()
