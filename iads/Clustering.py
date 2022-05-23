# -*- coding: utf-8 -*-

# Ben Kabongo B.
# Avril 2022

# Sorbonne Université
# LU3IN026 - Sciences des données
# Clustering


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist


def normalisation(in_df):
    arr = np.array(in_df)
    min_ = np.min(arr, axis=0)
    max_ = np.max(arr, axis=0)
    diff = max_ - min_
    diff[diff==0] = 1
    out = ((arr-min_)/diff)
    return pd.DataFrame(out, columns=in_df.columns)


def dist_euclidienne(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2) ** 2))


def dist_manhattan(arr1, arr2):
    return np.sum(np.abs(arr1-arr2))


def dist_vect(function, arr1, arr2):
    if function == 'euclidienne':
        return dist_euclidienne(arr1, arr2)
    if function == 'manhattan':
        return dist_manhattan(arr1, arr2)


def centroide(arr):
    return np.mean(arr, axis=0)


def dist_centroides(arr1, arr2):
    return dist_euclidienne(centroide(arr1), centroide(arr2))


def initialise(df):
    return {i:[i] for i in range(len(df))}


def fusionne(df, in_partition, verbose=False):
    dist_min = +np.inf
    k1_min, k2_min = -1, -1
    for k1, v1 in in_partition.items():
        for k2, v2 in in_partition.items():
            if k1 == k2:
                continue
            dist = dist_centroides(df.iloc[v1], df.iloc[v2])
            if dist < dist_min:
                dist_min = dist
                k1_min, k2_min = k1, k2
    out_partition = dict(in_partition)
    if k1_min != -1:
        del out_partition[k1_min]
        del out_partition[k2_min]
        out_partition[max(in_partition)+1] = [*in_partition[k1_min], *in_partition[k2_min]]
    if verbose:
        print(f'Distance mininimale trouvée entre  [{k1_min}, {k2_min}]  =  {dist_min}')
    return out_partition, k1_min, k2_min, dist_min


def clustering_hierarchique(df, verbose=False, dendrogramme=False):
    partition = initialise(df)
    results = []
    for _ in range(len(df)):
        partition, k1, k2, dist = fusionne(df, partition, verbose=verbose)
        results.append([k1, k2, dist, len(partition[max(partition.keys())])])
    results = results[:-1]
    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(results, leaf_font_size=24.)
        plt.show()
    return results


def dist_linkage_clusters(linkage, dist_func, arr1, arr2):
    r = cdist(arr1, arr2, dist_func)
    if linkage == 'complete':
        return np.max(r)
    if linkage == 'simple':
        return np.min(r)
    if linkage == 'average':
        return np.mean(r)


def fusionne_linkage(linkage, df, in_partition, dist_func='euclidean', verbose=False):
    dist_min = +np.inf
    k1_min, k2_min = -1, -1
    for k1, v1 in in_partition.items():
        for k2, v2 in in_partition.items():
            if k1 == k2:
                continue
            dist = dist_linkage_clusters(linkage, dist_func, df.iloc[v1], df.iloc[v2])
            if dist < dist_min:
                dist_min = dist
                k1_min, k2_min = k1, k2
    out_partition = dict(in_partition)
    if k1_min != -1:
        del out_partition[k1_min]
        del out_partition[k2_min]
        out_partition[max(in_partition)+1] = [*in_partition[k1_min], *in_partition[k2_min]]
        if verbose:
            print(f'Distance mininimale trouvée entre  [{k1_min}, {k2_min}]  =  {dist_min}')
    return out_partition, k1_min, k2_min, dist_min


def clustering_hierarchique_linkage(linkage, df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    partition = initialise(df)
    results = []
    for _ in range(len(df)):
        partition, k1, k2, dist = fusionne_linkage(linkage, df, partition, 
                                                   dist_func, verbose)
        results.append([k1, k2, dist, len(partition[max(partition.keys())])])
    results = results[:-1]
    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(results, leaf_font_size=24.)
        plt.show()
    return results


def clustering_hierarchique_complete(df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    return clustering_hierarchique_linkage('complete', df, dist_func,
                                          verbose, dendrogramme)


def clustering_hierarchique_simplee(df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    return clustering_hierarchique_linkage('simple', df, dist_func,
                                          verbose, dendrogramme)


def clustering_hierarchique_average(df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    return clustering_hierarchique_linkage('average', df, dist_func,
                                          verbose, dendrogramme)


def dist_vect(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


def inertie_cluster(cluster):
    return np.sum(dist_vect(cluster, centroide(cluster)) ** 2)


def init_kmeans(K, Ens):
    return np.array(random.sample(list(np.array(Ens)), K))


def plus_proche(x,C):
    return np.argmin(cdist(np.array(x).reshape(1, -1), C), axis=1)[0]


def affecte_cluster(X,C):
    Y = np.argmin(cdist(X, C), axis=1)
    return {c:list(np.where(Y==c)[0]) for c in range(len(C))}


def nouveaux_centroides(X,U):
    return np.array([np.mean(np.array(X)[idxs], axis=0) for idxs in U.values()])


def inertie_globale(X, U):
    return np.sum([inertie_cluster(np.array(X)[idxs]) for idxs in U.values()])


def kmoyennes(K, X, eps, iter_max, verbose=True):
    ig = 0
    U = {}
    C = init_kmeans(K, X)
    for i in range(iter_max):
        U = affecte_cluster(X, C)
        ig_ = inertie_globale(X, U)
        if verbose: print(f'iteration {i} Inertie : {ig_:.4f} Difference: {np.abs(ig_-ig):.4f}')
        if np.abs(ig_-ig) < eps:
            break
        ig = ig_
        C = nouveaux_centroides(X, U)
    return C, U


def distance_max_cluster(cluster):
    return np.max(cdist(cluster, cluster))


def co_dist(X, U):
    d = 0
    X = np.array(X)
    for idxs in U.values():
        d += distance_max_cluster(X[idxs])
    return d


co_inertie = inertie_globale


def index_dunn(X, U):
    return co_dist(X, U) / co_inertie(X, U)


def separabilite(C):
    A = cdist(C, C)
    A = A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
    return np.min(A)


def index_xie_beni(X, C, U):
    return co_inertie(X, U) / separabilite(C)
