from sklearn import cluster, metrics
from sklearn.decomposition import PCA

from plots import *
from constants import *


def scores_to_dataframe(scores):
    """Convert dictionary of evaluation scores to a dataframe"""
    res = pd.DataFrame(scores, index=NUM_CLUSTERS)
    res.index.name = "k"
    res = res.swaplevel(axis=1)
    return res


def cluster_(data, labels, method, n_clusters):
    """Test a given clustering algorithm for a given number of clusters"""
    if method not in CLUSTERING_METHODS:
        raise ValueError("Method not found: " + method)
        
    elif method == "kmeans":
        clus = cluster.KMeans(n_clusters=n_clusters, random_state=0)
        
    elif method == "agglomerative":
        clus = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity="manhattan", linkage="single")
        
    elif method == "spectral":
        clus = cluster.SpectralClustering(assign_labels="discretize", n_clusters=n_clusters, random_state=0)
    
    
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


def run_cluster_analysis(data, labels=None):
    """Run all cluster methods on the given data and return evaluation metrics"""
    method_scores = {}
    
    for method in CLUSTERING_METHODS:
        aris = []
        silhouettes = []
        dbs = []
        
        # For each method, we try
        # k={2, 3, ..., 6} number of clusters
        for k in NUM_CLUSTERS:
            ari_score, silhouette, db = cluster_(data, labels, method, k)
            
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

                
def optimize_ARI(X, y, n=100):
    """Visualize the best initial centroids for K-means,
    optimized for the ARI score; return all scores."""
    score = np.zeros((n, len(NUM_CLUSTERS)))
    
    # Compute scores for different random centroid initializations
    # and for n_clusters={2, 3, ..., 6}
    for j in range(score.shape[1]):
        for i in range(score.shape[0]):
            clus = cluster.KMeans(n_clusters=j+2, random_state=i)
            predicted = clus.fit_predict(X)
            score[i, j] = metrics.adjusted_rand_score(y, predicted)

    # Plot scores for each n_clusters
    for j in range(score.shape[1]):
        plt.plot(score[:, j])
    
    plt.legend(["2", "3", "4", "5", "6"])
    plt.ylim(bottom=0)
    plt.xlabel("Random state initializer")
    plt.ylabel("ARI score")
    plt.show()
    
    # Print results
    print(f"Max score: {score.max()}")
    print(f"Position of best score: {np.where(score == score.max())}")

    for k in range(score.shape[1]):
        print(f"Max ARI score for {k+2} clusters: {np.round(100 * score[:, k].max())}%")

    return score


def apply_pdx_centroids_on_patients(X_pdx_stdized_noctrl, y_pdx_noctrl, pats_log_stdized, state=116, dim=3, filename="labeled-patients-2d-scatter"):
    """Find best clusters on PDX data, apply those cluster centers on patient data.
    Clusters are fitted to standardized PDX data without controls."""
    # get optimal cluster
    clus = cluster.KMeans(n_clusters=3, random_state=state)
    predicted = clus.fit_predict(X_pdx_stdized_noctrl)
    ari_score = metrics.adjusted_rand_score(y_pdx_noctrl, predicted)
    print(f"Test ARI score: {ari_score}")
    print("Make sure the score == 1")
    
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
        data = pd.DataFrame(pats_components[:, :2], columns=["1st PC", "2nd PC"])
        data["predicted"] = patientLabels
        fig = px.scatter(data, x="1st PC", y="2nd PC", color="predicted")
        fig.show()        
        fig.write_image(f"{PLOT_DIR}/{filename}.png")
    return patientLabels