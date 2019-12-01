# -*- coding: utf-8 -*-
"""Various helper functions"""

import numpy as np
import pandas as pd

from load import load_genes_list
from constants import HORMONES


def df_to_tril(df):
    """Returns a lower triangular dataframe where entries
    above and including the main diagonal are set to zero.
    """
    for index, row in df.iterrows():
        row[index <= row.index] = 0
    return df


def gene_pairs_per_treatment():
    """Returns all pairs of genes differentially expressed
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
