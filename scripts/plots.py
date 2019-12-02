"""Plotting functions"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import chart_studio
chart_studio.tools.set_credentials_file(username='valentin.loftsson', api_key='SfH9Q8y2Cqzd38Sm0je4')
import chart_studio.plotly as py
import plotly.express as px

from constants import PLOT_DIR


def savefig(fig, filename):
    fig.savefig(f"{PLOT_DIR}/{filename}.png")


def plot_corr(corr, labels, filename="corr"):
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


def explained_variance_percentage(ratios):
    return round(sum(ratios) * 100, 2)


def pca_visualize_2d(data, labels=None, filename="pca_2d"):
    """Visualize data samples in 2D using first two principal components"""
    pca = PCA(n_components=3).fit(data)
    components = pca.transform(data)
    explained_var = explained_variance_percentage(pca.explained_variance_ratio_[:2])

    print(f"First 2 components explain {explained_var}% of the variance in the original data")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title("PCA visualization")
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")

    x = components[:, 0]
    y = components[:, 1]

    ax.scatter(x, y)
    
    # If we have labels, then we're plotting the PDX data
    if labels:        
        for i in range(len(y)):
            ax.annotate(labels[i], (x[i]+.5, y[i]+.5))
            
    plt.savefig(f"{PLOT_DIR}/{filename}.png")

    
def pca_visualize_3d(data, labels=None, filename="pats-pca-3d"):
    """Visualize data samples in 3D using first 3 principal components"""
    pca = PCA(n_components=3).fit(data)
    components = pca.transform(data)

    data = pd.DataFrame(components, columns=["1st PC", "2nd PC", "3rd PC"], index=labels).reset_index()
    
    explained_var = explained_variance_percentage(pca.explained_variance_ratio_)
    print(f"First 3 components explain {explained_var}% of the variance in the original data")
    
    if isinstance(labels, pd.MultiIndex):
        title="Expression levels in PDX subjects - PCA visualization"
        fig = px.scatter_3d(
            data,
            x="1st PC",
            y="2nd PC",
            z="3rd PC",
            hover_name="id",
            color="treatment",
            symbol="tumor",
            color_discrete_map={"ctrl": "blue", "dht": "red", "e2": "green", "p4": "yellow"},
            opacity=0.5,
            title=title
        )
    else:
        title="Expression levels in breast cancer patients - PCA visualization"
        fig = px.scatter_3d(
            data,
            x="1st PC",
            y="2nd PC",
            z="3rd PC",
            hover_name="index",
            opacity=0.5,
            title=title
        )

        fig.update_traces(
            marker=dict(
                size=4, color="rgb(17, 157, 255)", line=dict(width=2, color="rgb(231, 99, 250)")
            ),
            selector=dict(mode="markers"),
        )
            
    py.plot(fig, filename=filename)
    
    
def plot_pca_info(pca):
    y_pos = np.arange(len(pca.singular_values_))
    plt.bar(y_pos, pca.singular_values_, align="center", alpha=0.5)
    plt.ylabel("Values")
    plt.xlabel("Principal components")
    plt.title("PCA - Singular values")
    plt.show()

    y_pos = np.arange(len(pca.singular_values_))
    plt.bar(y_pos, pca.explained_variance_ratio_, align="center", alpha=0.5)
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