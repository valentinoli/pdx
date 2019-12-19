"""Various helper functions concerning clustering methods"""

from sklearn import cluster, metrics
from sklearn.decomposition import PCA

from plots import *
from constants import *


def scores_to_dataframe(scores):
    """Convert dictionary of evaluation scores to a dataframe"""
    res = pd.DataFrame(scores, index=NUM_CLUSTERS)
    res.index.name = "k"
    res = res.swaplevel(axis=1)
    res = round(res, 3)
    return res


def cluster_(data, labels, method, n_clusters, state):
    """Test a given clustering algorithm for a given number of clusters, with a provided initial state. Labels correspond to the actual datapoints' labels in the supervised clustering."""
    if method not in CLUSTERING_METHODS:
        raise ValueError("Method not found: " + method)
        
    elif method == "kmeans":
        clus = cluster.KMeans(n_clusters=n_clusters, random_state=state[method])

    elif method == "agglomerative":
        clus = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity="manhattan", linkage="single")
        
    elif method == "spectral":
        clus = cluster.SpectralClustering(assign_labels="discretize", eigen_tol=1e-10, n_clusters=n_clusters, random_state=state[method])
    
    
    # Predict cluster labels for each sample
    predicted = clus.fit_predict(data)
    
    # Compute performance metrics
    # 1. Adjusted rand score (pdx, supervised)
    # 2. Silhouette coefficient (patients, unsupervised)
    # 3. Davies-Bouldin score (patients, unsupervised)
    if labels is None:
        ari = None
        silhouette = metrics.silhouette_score(data, predicted, metric="euclidean")
        db = metrics.davies_bouldin_score(data, predicted)
    else:
        ari = metrics.adjusted_rand_score(labels, predicted)
        silhouette = None
        db = None        
    
    return ari, silhouette, db


def run_cluster_analysis(data, labels=None, random_state={"spectral": 0, "kmeans": 0}):
    """Run all cluster methods on the given data and return evaluation metrics in the form of a dict where keys are (method, evaluation metric) and values correspond to lists of the evaluation metric per k = 2 to 6 with the given method."""
    method_scores = {}
    
    for method in CLUSTERING_METHODS:
        aris = []
        silhouettes = []
        dbs = []
        
        # For each method, we try
        # k={2, 3, ..., 6} number of clusters
        for k in NUM_CLUSTERS:
            ari_score, silhouette, db = cluster_(data, labels, method, k, random_state)
            
            aris.append(ari_score)
            silhouettes.append(silhouette)
            dbs.append(db)
        
        if labels is None:
            # patients
            method_scores[method, "silhouette"] = silhouettes
            method_scores[method, "db"] = dbs
        else:
            # pdx
            method_scores[method, "ari"] = aris
    
    plot_analysis_results(method_scores)
    scores_df = scores_to_dataframe(method_scores)
    return scores_df

                
def optimize_ARI(X, y, method, n=100):
    """Visualize the best initial centroids for a clustering method,
    optimized for the ARI score; return optimum random state."""
    score = np.zeros((n, len(NUM_CLUSTERS)))
    
    # Compute scores for different random centroid initializations
    # and for n_clusters={2, 3, ..., 6}
    for j in range(score.shape[1]):
        for i in range(score.shape[0]):
            if method == "kmeans":
                clus = cluster.KMeans(n_clusters=j+2, random_state=i)
            elif method == "spectral":
                clus = cluster.SpectralClustering(assign_labels="discretize", n_clusters=j+2, random_state=i)
            else:
                raise ValueError("Method not found: " + method)
                
            predicted = clus.fit_predict(X)
            score[i, j] = metrics.adjusted_rand_score(y, predicted)

    # Plot scores for each n_clusters
    for j in range(score.shape[1]):
        plt.plot(score[:, j])
    
    plt.legend(list(NUM_CLUSTERS))
    plt.ylim(bottom=0)
    plt.xlabel("Random state initializer")
    plt.ylabel("ARI score")
    plt.show()
    
    optimum_state = np.where(score == score.max())[0][0]
    
    # Print results
    print(f"{method} clustering, max score: {score.max()}")
    print(f"Optimum random state resulting in the best ARI score: {optimum_state}")

    for k in range(score.shape[1]):
        print(f"Max ARI score for {k+2} clusters: {np.round(100 * score[:, k].max())}%")

    return optimum_state


def apply_pdx_centroids_on_patients(X_pdx_stdized_noctrl, y_pdx_noctrl, pats_log_stdized, state=116, dim=3, filename="labeled-patients-2d-scatter"):
    """Find best clusters on PDX data, apply those cluster centers on patient data.
    Clusters are fitted to standardized PDX data without controls."""
    # get optimal cluster
    clus = cluster.KMeans(n_clusters=3, random_state=state)
    predicted = clus.fit_predict(X_pdx_stdized_noctrl)
    ari_score = metrics.adjusted_rand_score(y_pdx_noctrl, predicted)
    
    # predict patient labels
    patientLabels = clus.predict(pats_log_stdized)
    pca = PCA()
    pats_components = pca.fit_transform(pats_log_stdized)
    if dim == 3:
        data = pd.DataFrame(pats_components[:, :3], columns=["1st PC", "2nd PC", "3rd PC"])
        data["predicted"] = patientLabels
        fig = px.scatter_3d(data, x="1st PC", y="2nd PC", z="3rd PC", color="predicted")
        fig.show()
    elif dim == 2:
        filename="labeled-patients"
        data = pd.DataFrame(pats_components[:, :2], columns=["1st PC", "2nd PC"])
        fig = sns.scatterplot(
            data["1st PC"],
            data["2nd PC"],
            hue=patientLabels,
            palette='Set1',
            s=100
        )
        plt.title("Labeled patient data")
        plt.xlabel("1st PC")
        plt.ylabel("2nd PC")
        plt.savefig(f"{PLOT_DIR}/{filename}.png")
    return patientLabels