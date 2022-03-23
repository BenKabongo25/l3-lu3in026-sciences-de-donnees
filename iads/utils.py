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


def plot_frontiere_V3(desc_set, label_set, w, kernel, step=30, forme=1, fname="out/tmp.pdf"):
    """ desc_set * label_set * array * function * int * int * str -> NoneType
        Note: le classifieur linéaire est donné sous la forme d'un vecteur de poids pour plus de flexibilité
    """
    # -----------
    # ETAPE 1: construction d'une grille de points sur tout l'espace défini par les points du jeu de données
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # -----------
    # Si vous avez du mal à saisir le concept de la grille, décommentez ci-dessous
    #plt.figure()
    #plt.scatter(grid[:,0],grid[:,1])
    #if True:
    #    return
    
    # -----------
    # ETAPE 2: calcul de la prediction pour chaque point de la grille
    res=np.array([kernel(grid[i,:])@w for i in range(len(grid)) ])
    # pour les affichages avancés, chaque dimension est présentée sous la forme d'une matrice
    res=res.reshape(x1grid.shape) 
    
    # -----------
    # ETAPE 3: le tracé
    #
    # CHOIX A TESTER en décommentant:
    # 1. lignes de contours + niveaux
    if forme <= 2 :
        fig, ax = plt.subplots() # pour 1 et 2
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
    if forme == 1:
        CS = ax.contour(x1grid,x2grid,res)
        ax.clabel(CS, inline=1, fontsize=10)
    #
    # 2. lignes de contour 0 = frontière 
    if forme == 2:
        CS = ax.contour(x1grid,x2grid,res, levels=[0], colors='k')
    #
    # 3. fonction de décision 3D
    if forme == 3 or forme == 4:
        #fig = plt.gcf()
        fig = plt.figure()
        ax = fig.gca(projection='3d') # pour 3 et 4
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
        ax.set_zlabel('f(X)')
    # 
    if forme == 3:
        surf = ax.plot_surface(x1grid,x2grid,res, cmap=cm.coolwarm)
    #
    # 4. fonction de décision 3D contour grid + transparence
    if forme == 4:
        norm = plt.Normalize(res.min(), res.max())
        colors = cm.coolwarm(norm(res))
        rcount, ccount, _ = colors.shape
        surf = ax.plot_surface(x1grid,x2grid,res, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
        surf.set_facecolor((0,0,0,0))
    
    # -----------
    # ETAPE 4: ajout des points
    negatifs = desc_set[label_set == -1]     # Ensemble des exemples de classe -1
    positifs = desc_set[label_set == +1]     # +1 
    # Affichage de l'ensemble des exemples en 2D:
    if forme <= 2:
        ax.scatter(negatifs[:,0],negatifs[:,1], marker='o', c='b') # 'o' pour la classe -1
        ax.scatter(positifs[:,0],positifs[:,1], marker='x', c='r') # 'x' pour la classe +1
    else:
        # on peut ajouter une 3ème dimension si on veut pour 3 et 4
        ax.scatter(negatifs[:,0],negatifs[:,1], -1, marker='o', c='b') # 'o' pour la classe -1
        ax.scatter(positifs[:,0],positifs[:,1], 1,  marker='x', c='r') # 'x' pour la classe +1
    
    # -----------
    # ETAPE 5 en 3D: régler le point de vue caméra:
    if forme == 3 or forme == 4:
        ax.view_init(20, 70) # a régler en fonction des données
    
    # -----------
    # ETAPE 6: sauvegarde (le nom du fichier a été fourni en argument)
    if fname != None:
        # avec les options pour réduires les marges et mettre le fond transprent
        plt.savefig(fname,bbox_inches='tight', transparent=True,pad_inches=0)


def create_XOR(n, sigma):
    one   = sigma * np.random.randn(n, 2) + [0, 0]
    two   = sigma * np.random.randn(n, 2) + [1, 1]
    three = sigma * np.random.randn(n, 2) + [0, 1]
    four  = sigma * np.random.randn(n, 2) + [1, 0]
    return (np.vstack((one, two, three, four)),
           np.hstack((-1*np.ones(2*n), np.ones(2*n))))


def crossval(X, Y, n, i):
    start, end = i*int(len(Y)/n), (i+1)*int(len(Y)/n)
    Xtrain = np.delete(X, np.s_[start:end], axis=0)
    Ytrain = np.delete(Y, np.s_[start:end], axis=0)
    Xtest = X[start:end]
    Ytest = Y[start:end]
    return Xtrain, Ytrain, Xtest, Ytest


def crossval_strat(X, Y, n, i):
    Xtrain1, Ytrain1, Xtest1, Ytest1 = crossval(X[Y==-1], Y[Y==-1], n, i)
    Xtrain2, Ytrain2, Xtest2, Ytest2 = crossval(X[Y==+1], Y[Y==+1], n, i)
    Xtrain = np.concatenate((Xtrain1, Xtrain2))
    Ytrain = np.concatenate((Ytrain1, Ytrain2))
    Xtest = np.concatenate((Xtest1, Xtest2))
    Ytest = np.concatenate((Ytest1, Ytest2))
    return Xtrain, Ytrain, Xtest, Ytest


