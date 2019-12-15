import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

from partial_dependence_plot import plot_ice, plot_pdp

dataset = pd.DataFrame(data=fetch_california_housing()['data'], columns=fetch_california_housing()['feature_names'])
features = dataset.values[:10000, :]
y = fetch_california_housing()['target'][:10000]

model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10)
model.fit(features, y)

# plot_df = pd.DataFrame()
# for f in range(len(dataset.columns)):
#     vals = plot_pdp(features=features, model=model, feature_number=f,
#                                         feature_name=dataset.columns[f])
#     plot_df[dataset.columns[f]] = vals
#     plot_df.plot(figsize=(16, 9))
#     plt.show()

plot_df = pd.DataFrame()
for f in range(len(dataset.columns)):
    res_df = plot_ice(features=features, model=model, feature_number=f, feature_name=dataset.columns[f])
    ax = res_df.plot(figsize=(16, 9), c='k', alpha=0.08, legend=False, linewidth=0.5)
    res_df.mean(axis=1).plot(c='red', linewidth=5, legend=False, ax=ax)
    plt.show()