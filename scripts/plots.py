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
    
def plot_pca_info(pca):
    y_pos = np.arange(len(pca.singular_values_))
    plt.bar(y_pos, pca.singular_values_, align="center", alpha=0.5)
    plt.ylabel("Values")
    plt.xlabel("Principal components")
    plt.title("PCA - Singular values")
    plt.show()

    y_pos = np.arange(len(pca.singular_values_))
    plt.bar(y_pos, pca.explained_variance_, align="center", alpha=0.5)
    plt.ylabel("Explained variance")
    plt.xlabel("Principal components")
    plt.title("PCA - explained variance")
    plt.show()
    
def plot_pca_expl_var(pca):
    fig, ax = plt.subplots()
    xi = np.arange(0, 33, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.3, 1.1)
    plt.plot(xi, y, marker="o", linestyle="--", color="b")

    plt.xlabel("Number of Components")
    plt.xticks(
        np.arange(0, 33, step=1)
    )  # change from 0-based array index to 1-based human-readable label
    plt.ylabel("Cumulative variance (%)")
    plt.title("The number of components needed to explain variance")

    plt.axhline(y=0.99, color="r", linestyle="-")
    plt.axhline(y=0.95, color="orange", linestyle="-")

    plt.text(0.5, 1, "99% cut-off threshold", color="red", fontsize=10)
    plt.text(0.5, 0.9, "95% cut-off threshold", color="orange", fontsize=10)

    ax.grid(axis="x")
    plt.show()
    
def plot_method_score(method, scores):
    x = np.arange(2,len(scores)+2)
    plt.plot(x, scores)
    plt.ylabel("Score")
    plt.xlabel("Number of clusters")
    plt.title(method + " score")
    plt.ylim(-0.5,1.0)
    plt.show()

def plot_method_silhouette(method, silhouettes):
    x = np.arange(2,len(silhouettes)+2)
    plt.plot(x, silhouettes)
    plt.ylabel("Silhouette")
    plt.xlabel("Number of clusters")
    plt.title(method + " silhouette")
    plt.ylim(-0.5,1.0)
    plt.show()