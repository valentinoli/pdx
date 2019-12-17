from sklearn import cluster, metrics
from sklearn.decomposition import PCA

from plots import *
from constants import *


def cluster_(data, labels, method, n_clusters, with_score):
    if method not in CLUSTERING_METHODS:
        raise ValueError("Method not found: " + method)
    elif method=='kmeans':
        clus = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    elif method=='agglomerative':
        clus = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='manhattan', linkage='single')
    elif method=='spectral':
        clus = cluster.SpectralClustering(assign_labels="discretize", n_clusters=n_clusters, random_state=0)
    elif method=='meanshift':
        clus = cluster.MeanShift()
    if(method=="spectral"):
        print("before clustering")
    predicted = clus.fit_predict(data)
    if(method=="spectral"):
                print("after clustering")
    if with_score:
        score = metrics.adjusted_rand_score(labels, predicted)
    else:
        score = None
    silhouette = metrics.silhouette_score(data, predicted, metric='euclidean')
    db = metrics.davies_bouldin_score(data, predicted)
    return score, silhouette, db


def test_all_methods(data, labels=None, with_score=False):
    print("REMINDER: Lower the DB index value, better is the clustering")
    values = {}
    for method in CLUSTERING_METHODS:
        print("Method: " + method)
        scores = []
        silhouettes = []
        dbs = []
        for k in range(2,8):
            score, silhouette, db = cluster_(data, labels, method, k, with_score)
            if with_score:
                scores.append(score)
            silhouettes.append(silhouette)
            dbs.append(db)
        values[method, 'db'] = dbs
        values[method, 'score'] = scores
        values[method, 'silhouette'] = silhouettes
        print("Scores: " + str(scores))
        print("DBS: " + str(dbs))
        print("Silhouettes: " + str(silhouettes))
    plot_index(values, with_score)

        
def test_agglomerative(data, labels):
    with_score = True
    if labels==None: with_score = False
    for affinity in AFFINITIES:
        for linkage in LINKAGES:
            if not(affinity!='euclidean' and linkage=='ward'):
                scores = []
                silhouettes = []
                for k in range(2,15):
                    clus = cluster.AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage)
                    predicted = clus.fit_predict(data)
                    if with_score:
                        score = metrics.adjusted_rand_score(labels, predicted)
                    else:
                        score = None
                    silhouette = metrics.silhouette_score(data, predicted, metric='euclidean')
                    scores.append(score)
                    silhouettes.append(silhouette)
                plot_method("agglomerative_"+affinity+"_"+linkage, with_scores, silhouettes, silhouettes, silhouettes, scores)

                
def optimize_ARI(X, Y, n=100):
    """Visualize the best initializer which is optimized for the ARI score, return all scores."""
    score = np.zeros((n,5))
    for j in range(score.shape[1]):
        for i in range(score.shape[0]):
            clus = cluster.KMeans(n_clusters=j+2, random_state=i)
            predicted = clus.fit_predict(X)
            score[i,j] = metrics.adjusted_rand_score(Y, predicted)

    for sc in range(score.shape[1]):
        plt.plot(score[:,sc])
    plt.legend(['2','3','4','5','6'])
    plt.ylim(bottom=0)
    plt.xlabel('Random state initializer')
    plt.ylabel('ARI score')
    plt.show()
    print("Max score: " + str(score.max()))
    print("Position of best score: " + str(np.where(score==score.max())))

    for k in np.arange(score.shape[1]):
        print("Max ARI score for " + str(k+2)+ " clusters: " + str(np.round(100*score[:,k].max()))+str('%'))
    
    return score

def applyClusterCentersOnPatients(X_pdx_stdized_noctrl, y_pdx_noctrl, pats_log_stdized):
    """Find best clusters on PDX data, apply those cluster centers on patient data. Clusters are fitted to standardized PDX data without controls."""
    # get optimal cluster
    clus = cluster.KMeans(n_clusters=3, random_state=116)
    predicted = clus.fit_predict(X_pdx_stdized_noctrl)
    print("Test ARI score: " + str(metrics.adjusted_rand_score(y_pdx_noctrl, predicted)))
    print("Make sure the score == 1")
    
    # predict patient labels
    patientLabels = clus.predict(pats_log_stdized)
    pca = PCA()
    pats_components = pca.fit_transform(pats_log_stdized)
    data = pd.DataFrame(pats_components[:,:3], columns=["1st PC", "2nd PC", "3rd PC"])
    data['predicted'] = patientLabels
    fig = px.scatter_3d(data, x="1st PC", y="2nd PC", z="3rd PC", color='predicted')
    fig.show()