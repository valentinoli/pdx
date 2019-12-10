from sklearn import cluster, metrics

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
    predicted = clus.fit_predict(data)
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
        scores = []
        silhouettes = []
        dbs = []
        for k in range(2,15):
            score, silhouette, db = cluster_(data, labels, method, k, with_score)
            if with_score:
                scores.append(score)
            silhouettes.append(silhouette)
            dbs.append(db)
        values[method, 'db'] = dbs
        values[method, 'score'] = scores
        values[method, 'silhouette'] = silhouettes
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
                