"""Constants"""

HORMONES = ["dht", "e2", "p4"]
UP_DOWN = ["up", "down"]
LABELS = dict(zip(HORMONES, [0, 1, 2]))
CORR_THRESHOLD = 0.6

# paths
DATA_DIR = "../data"

GENES_PATH = f"{DATA_DIR}/pdx/List of Genes Differentially Expressed upon Different Treatments.xlsx"
PDX_PATH = f"{DATA_DIR}/pdx/Human_matrix_DESEQ2normalized_removedlowlyexpressedgenes.xlsx"
PATIENTS_PATH = f"{DATA_DIR}/patients/TCGA_ALL_Samples_log_Normalized GS.xlsx"
PATIENTS_PATH_2 = (f"{DATA_DIR}/patients/BRCA.rnaseqv2__illuminahiseq_rnaseqv2__"
                   "unc_edu__Level_3__RSEM_genes_normalized__data.data.txt")