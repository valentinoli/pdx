# -*- coding: utf-8 -*-
"""Various helper functions"""
import pandas as pd

# paths
GENES_PATH = "../data/pdx/List of Genes Differentially Expressed upon Different Treatments.xlsx"
PDX_PATH = "../data/pdx/Human_matrix_DESEQ2normalized_removedlowlyexpressedgenes.xlsx"
PATIENTS_PATH = "../data/patients/TCGA_ALL_Samples_log_Normalized GS.xlsx"
PATIENTS_PATH_2 = ("../data/patients/BRCA.rnaseqv2__illuminahiseq_rnaseqv2__"
                   "unc_edu__Level_3__RSEM_genes_normalized__data.data.txt")

"""("../data/patients/gdac.broadinstitute.org_BRCA.Merge_rnaseqv2__"
    "illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__"
    "data.Level_3.2016012800.0.0.tar.gz")"""


def load_genes():
    """Load raw information about the genes"""
    hormones = ["dht", "e2", "p4"]
    up_down = ["up", "down"]
    columns = [(i, j) for i in hormones for j in up_down]
    multi_index = pd.MultiIndex.from_tuples(columns)

    return pd.read_excel(
        GENES_PATH,
        names=multi_index,
        skiprows=[0, 1],
        usecols=[1, 2, 4, 5, 7, 8],
    )


def load_genes_list():
    """Load the preprocessed list of genes"""
    genes_list = pd.read_excel(GENES_PATH, sheet_name=1)
    genes_list.columns = genes_list.columns.str.lower()
    return genes_list


def load_pdx(genes):
    """Load PDX tumor data, only retaining certain genes"""
    pdx = pd.read_excel(
        PDX_PATH,
        index_col=0,
        usecols=[1] + list(range(4, 46)),
    )
    
    return pdx.transpose()[genes]


def load_patients(genes):
    """Load patient data (part 1), only retaining certain genes"""
    patients = pd.read_excel(
        PATIENTS_PATH,
        index_col=0,
    )
    
    return patients.transpose()[genes]


def load_patients2(genes):
    """Load patient data (part 1), only retaining certain genes"""
    patients = pd.read_csv(
        PATIENTS_PATH_2,
        sep="\t",
        index_col=0,
        skiprows=1,
    )
    
    # remove number identifiers following the gene name
    patients.index = patients.index.str.split("|").map(lambda x: x[0])
    
    return patients.transpose()[genes]