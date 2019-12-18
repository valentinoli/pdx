# -*- coding: utf-8 -*-
"""Various helper functions"""

import numpy as np
import pandas as pd

from load import load_genes_list
from constants import HORMONES, TUMORS, LABELS_CTRL, LABELS_CTRL_INVERTED


def df_to_tril(df):
    """Return a lower triangular dataframe where entries
    above and including the main diagonal are set to zero.
    """
    for index, row in df.iterrows():
        row[index <= row.index] = 0
    return df


def gene_pairs_per_treatment():
    """Return all pairs of genes differentially expressed
    upon the same treatment as a Series object with the index
    being the pair and the value being the list of treatments"""
    genes_list = load_genes_list()
    series = []

    for h in HORMONES:
        # get all genes expressing hormone h
        genes_h = genes_list[genes_list[h]]
        glist = list(genes_h.genes)

        num_genes = len(glist)

        # Compute pairs of indices for the lower triangular part of a matrix
        # of size (num_genes x num_genes), excluding the diagonal
        # (we don't want to pair the genes to themselves)
        tril_indices = np.tril_indices(num_genes, k=-1)
        index_pairs = list(zip(tril_indices[0], tril_indices[1]))

        # Map the list of index-pairs to all possible pairs of genes
        pairs = [(glist[pair[0]], glist[pair[1]]) for pair in index_pairs]

        # Create the series and append to list
        series_h = pd.Series(data=h, index=pairs, name="pdx_hormone")
        series.append(series_h)

    # Concatenate the list of series 
    genes_pairs = pd.concat(series, sort=False).groupby(level=0).apply(list)
    return genes_pairs


def df_standardize_columns(df):
    """Standardize columns of a dataframe"""
    return (df-df.mean(axis=0)) / df.std(axis=0)


def pdx_standardize(X_pdx):
    """Standardize the PDX feature matrix per tumor and return the concatenated dataframe,
     the aim being to eliminate bias introduced by different tumors being injected"""
    dfs_stdized = []
    for tumor in TUMORS:
        df = X_pdx.xs(tumor, level=1, drop_level=False)
        df_stdized = df_standardize_columns(df)
        dfs_stdized.append(df_stdized)
    
    return pd.concat(dfs_stdized).sort_values(["treatment", "tumor"])

def describe_prediction(predicted, actual, with_ctrl=True):
    """Return hormonal composition of the found clusters."""
    for cluster in np.unique(actual):
        print("Cluster %d contains:" % cluster)
        contains = actual[predicted==cluster]
        for label in np.unique(contains):
            count = np.count_nonzero(contains==label)
            if(with_ctrl):
                label_name = LABELS_CTRL_INVERTED[label]
            else:
                label_name = LABELS_INVERTED[label]
            print("%d %s samples" % (count,label_name))
        print("")
        
def get_gene_ratios(data, labels, ctrl_index=0):
    unique_labels = np.unique(labels)
    output = np.zeros((unique_labels.shape[0],data.shape[1]))
    for label in unique_labels:
        output[label] = np.mean(data[labels==label], axis=0)
    output = output / output[ctrl_index,:]
    return pd.DataFrame(data=output, columns=data.columns)