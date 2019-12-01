"""Plotting functions"""

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from constants import PLOT_DIR


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
    fig.savefig(f"{PLOT_DIR}/{filename}.png")
    
    
def plot_feature_distributions(data, filename="feature_distribution", ylim=[0, 5000], boxes=13):
    """Plots the distribution of features on boxplots"""
    names = data.columns
    num_plots = len(names)
    rows = math.ceil(num_plots / boxes)
    fig, ax = plt.subplots(rows, 1, figsize=(15, 50))

    for i in range(0, num_plots, boxes):
        index = range(i, min(i+boxes, num_plots))
        ax[i//boxes].boxplot(x=data.iloc[:, index].T, labels=names[index])
        ax[i//boxes].set_ylim(ylim)

    fig.savefig(f"{PLOT_DIR}/{filename}.png")
    

def plot_means_std_patients(pat, pat2, filename="patients_mean_std"):
    """Plot means and stds of patient datasets"""
    fig, ax = plt.subplots(figsize=(10, 40))
    
    labels = pat["mean"].index
    ind = np.arange(len(labels))
    barheight = 0.35

    ax.barh(y=ind, width=pat["mean"].values, height=barheight, color='r', xerr=pat["std"].values)
    ax.barh(y=ind + barheight, width=pat2["mean"].values, height=barheight, color='y', xerr=pat2["std"].values)
    ax.set_yticks(ind + barheight / 2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.title("Feature means and stds for patient-datasets")
    plt.xlabel("mean and standard deviation")
    plt.xlim(0, 20000)
    plt.legend(("first", "second"))

    plt.savefig(f"{PLOT_DIR}/{filename}.png")
    
    