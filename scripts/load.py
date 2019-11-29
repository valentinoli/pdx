# -*- coding: utf-8 -*-
"""Module for loading and manipulating the data"""

import pickle
import pandas as pd
from constants import *


def load_genes():
    """Load raw information about the genes"""
    pkl = f"{DATA_DIR}/genes_raw.pkl"
    try:
        return pd.read_pickle(pkl)
    
    except:
        columns = [(i, j) for i in HORMONES for j in UP_DOWN]
        multi_index = pd.MultiIndex.from_tuples(columns)

        genes = pd.read_excel(
            GENES_PATH,
            names=multi_index,
            skiprows=[0, 1],
            usecols=[1, 2, 4, 5, 7, 8],
        )
        genes.to_pickle(pkl)
        return genes


def load_genes_list():
    """Load the preprocessed list of genes"""
    pkl = f"{DATA_DIR}/genes_list.pkl"
    try:
        return pd.read_pickle(pkl)
    
    except:
        genes_list = pd.read_excel(GENES_PATH, sheet_name=1)
        genes_list.columns = genes_list.columns.str.lower()
        genes_list.to_pickle(pkl)
        return genes_list

    
def load_patients():
    """Load patient data (part 1), only retaining certain genes"""
    pkl = f"{DATA_DIR}/patients.pkl"
    try:
        return pd.read_pickle(pkl)
    
    except:
        genes = load_genes_list().genes
        
        patients = pd.read_excel(
            PATIENTS_PATH,
            index_col=0,  # first column contains genes
        ).T
        
        # replace indexing of patients with RangeIndex
        patients.reset_index(drop=True, inplace=True)
        
        # keep only selected genes
        patients = patients.loc[:, patients.columns.isin(genes)]
        
        patients.to_pickle(pkl)
        return patients


def load_patients2():
    """Load patient data (part 1), only retaining certain genes"""
    pkl = f"{DATA_DIR}/patients2.pkl"
    try:
        return pd.read_pickle(pkl)
    
    except:
        genes = load_genes_list().genes
        
        patients = pd.read_csv(
            PATIENTS_PATH_2,
            sep="\t",
            index_col=0,  # first column contains genes
            skiprows=1,
        ).T
        
        # Replace indexing of patients with RangeIndex
        patients.reset_index(drop=True, inplace=True)

        # Remove number identifiers that follow the gene name
        patients.columns = patients.columns.str.split("|").map(lambda x: x[0])

        # Remove the default name
        patients.columns.set_names(None, inplace=True)
        
        # Keep only selected genes
        patients = patients.loc[:, patients.columns.isin(genes)]
        
        patients.to_pickle(pkl)
        return patients
    

def load_pdx():
    """Load PDX tumor data, only retaining certain genes"""
    pkl = f"{DATA_DIR}/pdx.pkl"
    try:
        return pd.read_pickle(pkl)
    
    except:
        genes = load_genes_list().genes
        patients = load_patients()
        
        pdx = pd.read_excel(
            PDX_PATH,
            index_col=0,
            usecols=[1] + list(range(4, 46)),
        ).T

        # Not all the genes are expressed in every cell
        # and within a broad set of samples, so we discard
        # genes not expressed in the patients datasets
        pdx = pdx.loc[:, pdx.columns.isin(genes) & pdx.columns.isin(patients.columns)]
        
        # Remove subjects exposed to two treatments
        pdx = pdx[~pdx.index.str.contains(r"\+")]
                
        
        index_split = pdx.index.str.lower().str.rsplit("_", 1)
        
        # Transform the index into a multi-index
        # First level:  treatments (dht, e2, p4, ctrl)
        # Second level: subject id
        pdx.index = (
            index_split
            .map(lambda x: (x[1], x[0]))
            .set_names(["treatment", "id"])
        )
        
        # Sort gene names alphabetically
        # since they are not sorted beforehand
        pdx.columns = pdx.columns.sort_values().set_names(None)
        
        # Add label columns
        pdx.insert(0, "label", index_split.map(lambda x: LABELS_CTRL[x[1]]))
        
        # Sort by label
        pdx = pdx.sort_values("label")
        
        pdx.to_pickle(pkl)    
        return pdx
