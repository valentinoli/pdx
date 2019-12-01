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
    return score, silhouette

def test_all_methods(data, labels):
    with_score = True
    if labels.empty: with_score = False
    for method in CLUSTERING_METHODS:
        scores = []
        silhouettes = []
        for k in range(2,15):
            score, silhouette = cluster_(data, labels, method, k, with_score)
            scores.append(score)
            silhouettes.append(silhouette)
        if with_score:
            plot_method_score(method, scores)
        plot_method_silhouette(method, silhouettes)

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
                if with_score:
                    plot_method_score(method, scores)
                plot_method_silhouette("agglomerative_"+affinity+"_"+linkage, silhouettes)