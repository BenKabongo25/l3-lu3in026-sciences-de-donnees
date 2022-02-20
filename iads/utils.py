# -*- coding: utf-8 -*-

# Ben Kabongo B.
# Février 2022

# Sorbonne Université
# LU3IN026 - Sciences des données
# Fonctions utiles

import numpy as np
import matplotlib.pyplot as plt


def genere_dataset_uniform(p, n, xmin, xmax):
    """
    génère un dataset uniforme
    """
    return (np.random.uniform(xmin, xmax, (2*n, p)),
           np.array([-1 for _ in range(n)] + [1 for _ in range(n)]))


def genere_dataset_gaussian(pos_center, pos_sigma, neg_center, neg_sigma, nb_points):
    """
    génère un dataset gaussien
    """
    return (np.vstack((np.random.multivariate_normal(neg_center, neg_sigma, nb_points),
                       np.random.multivariate_normal(pos_center, pos_sigma, nb_points))),
            np.hstack((-1*np.ones(nb_points), np.ones(nb_points))))


def plot2DSet(data_desc, data_label):
    """
    affiche les points des données en fonction des labels
    :param data_desc: points des données
    :param data_label: étiquettes des données
    """
    data_negatifs = data_desc[data_label == -1]
    data_positifs = data_desc[data_label == +1]
    plt.scatter(data_negatifs[:,0], data_negatifs[:,1], marker='o', color='red')
    plt.scatter(data_positifs[:,0], data_positifs[:,1], marker='x', color='blue')


def plot_frontiere(desc_set, label_set, classifier, step=200):
    """
    affiche la frontière de décision associée au classifieur
    :param desc_set: données
    :param label_set: étiquettes des données
    :param classifier: classifieur
    :param step
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))

    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])


def create_XOR(n, sigma):
    one   = sigma * np.random.randn(n, 2) + [0, 0]
    two   = sigma * np.random.randn(n, 2) + [1, 1]
    three = sigma * np.random.randn(n, 2) + [0, 1]
    four  = sigma * np.random.randn(n, 2) + [1, 0]
    return (np.vstack((one, two, three, four)),
           np.hstack((-1*np.ones(2*n), np.ones(2*n))))
