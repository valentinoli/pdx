"""Constants"""

UP_DOWN = ["up", "down"]

CTRL = "ctrl"
HORMONES = ["dht", "e2", "p4"]
HORMONES_CTRL = HORMONES + [CTRL]
TUMORS = ["t111", "t110", "pl015"]

LABELS = dict(zip(HORMONES, list(range(3))))
LABELS_CTRL = dict(zip(HORMONES_CTRL, list(range(4))))

LABELS_INVERTED = dict(map(reversed, LABELS.items()))
LABELS_CTRL_INVERTED = dict(map(reversed, LABELS_CTRL.items()))

CORR_THRESHOLD = .6

# paths
DATA_DIR = "../data"
PLOT_DIR = DATA_DIR + "/plot"
PKL_DIR = DATA_DIR + "/pickle"

GENES_PATH = f"{DATA_DIR}/pdx/List of Genes Differentially Expressed upon Different Treatments.xlsx"
PDX_PATH = f"{DATA_DIR}/pdx/Human_matrix_DESEQ2normalized_removedlowlyexpressedgenes.xlsx"
PATIENTS_PATH = f"{DATA_DIR}/patients/TCGA_ALL_Samples_log_Normalized GS.xlsx"
PATIENTS_PATH_2 = (f"{DATA_DIR}/patients/BRCA.rnaseqv2__illuminahiseq_rnaseqv2__"
                   "unc_edu__Level_3__RSEM_genes_normalized__data.data.txt")

CLUSTERING_METHODS = ['agglomerative', 'kmeans', 'spectral', 'meanshift']
CLUSTERING_INDEXING_METHODS = ['db', 'score', 'silhouette']
AFFINITIES = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
LINKAGES = ['ward', 'average', 'complete', 'single']
