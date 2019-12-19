"""Plotting functions"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import chart_studio
chart_studio.tools.set_credentials_file(
    username="valentin.loftsson",
    api_key="SfH9Q8y2Cqzd38Sm0je4"
)
import chart_studio.plotly as py
import plotly.express as px

from constants import *


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
        row = i//boxes
        ax[row].boxplot(x=data.iloc[:, index].T, labels=names[index])
        ax[row].set_ylim(ylim)
    
        ax[row].set_ylabel("RNA expression level")
        
    plt.xlabel("gene")
    fig.savefig(f"{PLOT_DIR}/{filename}.png")
    

def plot_means_std_patients(pat, pat2, filename="patients_mean_std"):
    """Plot means and stds of patient datasets"""
    fig, ax = plt.subplots(figsize=(10, 40))
    
    labels = pat["mean"].index
    ind = np.arange(len(labels))
    barheight = 0.35

    ax.barh(y=ind, width=pat["mean"].values, height=barheight, color="r", xerr=pat["std"].values)
    ax.barh(y=ind + barheight, width=pat2["mean"].values, height=barheight, color="y", xerr=pat2["std"].values)
    ax.set_yticks(ind + barheight / 2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    
    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True
    plt.title("Feature means and stds for patient-datasets")
    plt.xlabel("mean and standard deviation")
    plt.xlim(0, 20000)
    plt.legend(("first", "second"))

    plt.savefig(f"{PLOT_DIR}/{filename}.png")


def sum_ratios_to_percentage(ratios):
    """Sums ratios and converts to two-decimal rounded percentage"""
    return round(sum(ratios) * 100, 2)


def pca_visualize_2d(data, index=None, filename="pca_2d", title="PCA visualization"):
    """Visualize data samples in 2D using first two principal components"""
    pca = PCA(n_components=3).fit(data)
    components = pca.transform(data)
    explained_var = sum_ratios_to_percentage(pca.explained_variance_ratio_[:2])

    print(f"First 2 components explain {explained_var}% of the variance in the original data")

    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")

    x = components[:, 0]
    y = components[:, 1]

    if isinstance(index, pd.MultiIndex):
        sns.scatterplot(
            x,
            y,
            hue=index.get_level_values(0),
            style=index.get_level_values(1),
            s=100,
        )
        
        ax.legend(bbox_to_anchor=(1.05, 1.025))
    else:
        sns.scatterplot(x, y)
       
    ax.grid(True)
    plt.savefig(f"{PLOT_DIR}/{filename}.png")

    
def pca_visualize_3d(data, labels=None, filename="pats-pca-3d"):
    """Visualize data samples in 3D using first 3 principal components"""
    pca = PCA(n_components=3).fit(data)
    components = pca.transform(data)

    data = pd.DataFrame(components, columns=["1st PC", "2nd PC", "3rd PC"], index=labels).reset_index()
    
    explained_var = sum_ratios_to_percentage(pca.explained_variance_ratio_)
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
    
    
def plot_pca_expl_var(pca, steps=33):
    """Plot cumulative explained variance of principal components"""
    fig, ax = plt.subplots()
    fig.set_size_inches(steps/4, 4)
    xi = np.arange(0, steps, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.3, 1.1)
    plt.plot(xi, y, marker="o", linestyle="--", color="b")

    plt.xlabel("Number of Components")
    plt.xticks(
        np.arange(0, steps, step=1)
    )  # change from 0-based array index to 1-based human-readable label
    plt.ylabel("Cumulative variance (%)")
    plt.title("The number of components needed to explain variance")

    plt.axhline(y=0.99, color="r", linestyle="-")
    plt.axhline(y=0.95, color="orange", linestyle="-")

    plt.text(0.5, 1, "99% cut-off threshold", color="red", fontsize=10)
    plt.text(0.5, 0.9, "95% cut-off threshold", color="orange", fontsize=10)

    ax.grid(axis="x")
    plt.show()
    
    
def plot_analysis_results(scores):
    """Plot computed evaluation metrics for each clustering method"""
    keys = np.array(list(scores.keys()))
    methods = np.unique(keys[:, 0])
    metrics = np.unique(keys[:, 1])
    
    nrows = len(metrics)
    ncols = len(methods)
    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    fig.set_figwidth(5 * ncols)
    fig.set_figheight(4.75 * nrows)
    
    for i, clus_metric in enumerate(metrics):
        for j, clus_method in enumerate(methods):
            vals = scores[clus_method, clus_metric]
            
            ax[i, j].plot(list(NUM_CLUSTERS), vals)
            ax[i, j].set_ylabel(clus_metric)
            ax[i, j].set_xlabel("num_clusters")
            ax[i, j].set_title(f"{clus_method}_{clus_metric}")
                
            if clus_metric == "db":
                ax[i, j].set_ylim(0.0, 3.5)
            else:
                ax[i, j].set_ylim(-0.5, 1.0)
            
                    
def pca_gene_composition(data, filename="pca_gene_composition", title="PCA gene composition"):
    """Visualize which genes contribute to the first 3 principal components."""
    pca = PCA(n_components=3).fit(data)
    plt.matshow(pca.components_, cmap="magma")
    plt.yticks([0, 1, 2], ["1st PC", "2nd PC", "3rd PC"], fontsize=8)
    plt.colorbar()
    plt.xticks(range(len(data.columns.values)), data.columns.values, rotation=80, ha="left", fontsize=7)
    plt.savefig(f"{PLOT_DIR}/{filename}.svg")
    plt.show()
    
    
def plot_kmeans_patient(data, predicted, dim=3, title="K-means labeled patient data", filename="kmeans-labeled-patients"):
    """Visualize K-means clustering using PCA"""
    if dim == 2:
        fig = sns.scatterplot(data["1st PC"], data["2nd PC"], hue=predicted, palette='Set1', s=100)
        plt.title(title)
        plt.xlabel("1st PC")
        plt.ylabel("2nd PC")
        plt.savefig(f"{PLOT_DIR}/{filename}.png")
    elif dim == 3:
        data['predicted'] = predicted
        fig = px.scatter_3d(data, x="1st PC", y="2nd PC", z="3rd PC", color='predicted', opacity=0.5,title=title)
        py.plot(fig, filename=filename)
 
