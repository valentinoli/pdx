"""Plotting functions"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_corr(corr, labels, filename='corr'):
    """Plots the lower triangular absolute correlation matrix on a heatmap"""
    tril_abs_corr = np.abs(np.tril(corr, k=-1))

    plt.figure(figsize=(25, 25))

    heatmap = sns.heatmap(
        tril_abs_corr,
        square=True,
        linewidths=.005,
        xticklabels=labels,
        yticklabels=labels,
        mask=(tril_abs_corr == 0)  # mask cells with missing values
    )
    fig = heatmap.get_figure()
    fig.savefig(f"../data/{filename}.png")
    
    
def plot_feature_distributions(data, names, filename="feature_distribution"):
    """Plots the distribution of features on boxplots"""
    step=10
    fig, ax = plt.subplots(step, 1, figsize=(15, 50))

    for i in range(0, len(names), step):
        index = range(i, min(i+step, len(names)))
        ax[i//10].boxplot(x=data.iloc[:, index].T, labels=names[index])
        ax[i//10].set_ylim([0, 5000])

    fig.savefig(f"../data/{filename}.png")
    
