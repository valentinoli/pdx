# -*- coding: utf-8 -*-
"""Various helper functions"""
import pickle
import pandas as pd

# paths
GENES_PATH = "../data/pdx/List of Genes Differentially Expressed upon Different Treatments.xlsx"
PDX_PATH = "../data/pdx/Human_matrix_DESEQ2normalized_removedlowlyexpressedgenes.xlsx"
PATIENTS_PATH = "../data/patients/TCGA_ALL_Samples_log_Normalized GS.xlsx"
PATIENTS_PATH_2 = ("../data/patients/BRCA.rnaseqv2__illuminahiseq_rnaseqv2__"
                   "unc_edu__Level_3__RSEM_genes_normalized__data.data.txt")

GENES_RAW_PATH_PKL = "../data/genes_raw.pkl"
GENES_PATH_PKL = "../data/genes.pkl"
PDX_PATH_PKL = "../data/pdx.pkl"
PATIENTS_PATH_PKL = "../data/patients.pkl"
PATIENTS_PATH_2_PKL = "../data/patients2.pkl"

"""("../data/patients/gdac.broadinstitute.org_BRCA.Merge_rnaseqv2__"
    "illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__"
    "data.Level_3.2016012800.0.0.tar.gz")"""


def load_genes():
    """Load raw information about the genes"""
    try:
        return pd.read_pickle(GENES_RAW_PATH_PKL)
    
    except:
        hormones = ["dht", "e2", "p4"]
        up_down = ["up", "down"]
        columns = [(i, j) for i in hormones for j in up_down]
        multi_index = pd.MultiIndex.from_tuples(columns)

        genes = pd.read_excel(
            GENES_PATH,
            names=multi_index,
            skiprows=[0, 1],
            usecols=[1, 2, 4, 5, 7, 8],
        )
        genes.to_pickle(GENES_RAW_PATH_PKL)
        return genes


def load_genes_list():
    """Load the preprocessed list of genes"""
    try:
        return pd.read_pickle(GENES_PATH_PKL)
    
    except:
        genes_list = pd.read_excel(GENES_PATH, sheet_name=1)
        genes_list.columns = genes_list.columns.str.lower()
        genes_list.to_pickle(GENES_PATH_PKL)
        return genes_list

    
def load_patients():
    """Load patient data (part 1), only retaining certain genes"""
    try:
        return pd.read_pickle(PATIENTS_PATH_PKL)
    
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
        
        patients.to_pickle(PATIENTS_PATH_PKL)
        return patients


def load_patients2():
    """Load patient data (part 1), only retaining certain genes"""
    try:
        return pd.read_pickle(PATIENTS_PATH_2_PKL)
    
    except:
        genes = load_genes_list().genes
        
        patients = pd.read_csv(
            PATIENTS_PATH_2,
            sep="\t",
            index_col=0,  # first column contains genes
            skiprows=1,
        ).T
        
        # replace indexing of patients with RangeIndex
        patients.reset_index(drop=True, inplace=True)

        # remove number identifiers that follow the gene name
        patients.columns = patients.columns.str.split("|").map(lambda x: x[0])
        
        # keep only selected genes
        patients = patients.loc[:, patients.columns.isin(genes)]
        
        patients.to_pickle(PATIENTS_PATH_2_PKL)
        return patients
    

def load_pdx():
    """Load PDX tumor data, only retaining certain genes"""
    try:
        return pd.read_pickle(PDX_PATH_PKL)
    
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

        # Remove control subjects and subjects exposed to two treatments
        pdx = pdx[~pdx.index.str.contains(r"\+|CTRL")]
        
        index_split = pdx.index.str.lower().str.rsplit("_", 1)

        # Transform the index into a multiindex
        # First level:  treatments (p4, dht, e2)
        # Second level: subject id
        pdx.index = (
            index_split
            .map(lambda x: (x[1], x[0]))
            .set_names(['treatment', 'id'])
        )
        
        # Sort gene names alphabetically
        # since they are not sorted beforehand
        pdx.columns = pdx.columns.sort_values().set_names(None)
        
        # Add label
        labels = {"dht": 0, "p4": 1, "e2": 2}
        pdx.insert(0, "label", index_split.map(lambda x: labels[x[1]]))
        
        # Sort by label
        pdx = pdx.sort_values("label")
        
        pdx.to_pickle(PDX_PATH_PKL)    
        return pdx
