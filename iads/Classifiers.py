# -*- coding: utf-8 -*-

# Ben Kabongo B.
# Février 2022

# Sorbonne Université
# LU3IN026 - Sciences des données
# Classifieurs


import numpy as np


class Classifier:
    """
    Classe de base des classifieurs
    """

    def __init__(self, input_dimension: int):
        """
        :param input_dimension (int) : dimension de la description des exemples
        Hypothèse : input_dimension > 0
        """
        assert input_dimension > 0
        self.input_dimension = input_dimension

    def train(self, desc_set: np.ndarray, label_set: np.ndarray):
        """
        Permet d'entrainer le modele sur l'ensemble donné
        :param desc_set: ndarray avec des descriptions
        :param label_set: ndarray avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x: np.ndarray) -> float:
        """
        :param x: une description
        :return rend le score de prédiction sur x (valeur réelle)
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x: np.ndarray) -> int:
        """
        :param x: une description
        :return rend la prediction sur x (soit -1 ou soit +1)
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set: np.ndarray, label_set: np.ndarray) -> float:
        """
        Permet de calculer la qualité du système sur un dataset donné
        :param desc_set: ndarray avec des descriptions
        :param label_set: ndarray avec les labels correspondants
        :return la performance du classifieur
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        return np.sum(np.array([self.predict(x) for x in desc_set]) == label_set) / len(label_set)


class ClassifierLineaireRandom(Classifier):
    """
    Classifieur linéaire aléatoire
    """

    def __init__(self, input_dimension):
        Classifier.__init__(self, input_dimension)
        v = np.random.uniform(-1, 1, input_dimension)
        self.w = v / np.linalg.norm(v)

    def train(self, desc_set, label_set):
        pass

    def score(self,x):
        return np.dot(self.w, x)

    def predict(self, x):
        return -1 if self.score(x) <= 0 else 1


class ClassifierKNN(Classifier):
    """
    Classifieur par K plus proches voisins.
    """

    def __init__(self, input_dimension: int, k: int):
        """
        :param input_dimension (int) : dimension d'entrée des exemples
        :param k (int) : nombre de voisins à considérer
        Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.k = k
        self.desc_set = None
        self.label_set = None

    def score(self,x):
        """
        :param x: une description : un ndarray
        :return rend la proportion de +1 parmi les k ppv de x (valeur réelle)
        """
        dist = np.linalg.norm(self.desc_set-x, axis=1)
        argsort = np.argsort(dist)
        score = np.sum(self.label_set[argsort[:self.k]] == 1)
        return 2 * (score/self.k -.5)

    def predict(self, x):
        return -1 if self.score(x)/2 + .5 <= .5 else +1

    def train(self, desc_set, label_set):
        self.desc_set = desc_set
        self.label_set = label_set


class ClassifierKNN_MC(Classifier):
    """
    Classifieur KNN multi-classe
    """

    def __init__(self, input_dimension, k, nb_class):
        """
        :param input_dimension (int) : dimension d'entrée des exemples
        :param k (int) : nombre de voisins à considérer
        :param nb_class (int): nombre de classes
        Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.k = k
        self.nb_class = nb_class
        self.data_set = None
        self.label_set = None

    def train(self, data_set, label_set):
        self.data_set = data_set
        self.label_set = label_set

    def score(self, x):
        dist = np.linalg.norm(self.data_set-x, axis=1)
        argsort = np.argsort(dist)
        classes = self.label_set[argsort[:self.k]]
        uniques, counts = np.unique(classes, return_counts=True)
        return uniques[np.argmax(counts)]/self.nb_class

    def predict(self, x):
        return self.score(x)*self.nb_class


class ClassifierPerceptron(Classifier):
    """
    Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate, init=0):
        """
        :param input_dimension (int) : dimension de la description des exemples (>0)
        :param learning_rate : epsilon
        :param init est le mode d'initialisation de w:
            - si 0 (par défaut): initialisation à 0 de w,
            - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        if init == 0:
            self.w = np.zeros(input_dimension)
        else:
            self.w = .001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)

    def train_step(self, desc_set: np.ndarray, label_set: np.ndarray):
        """
        Réalise une unique itération sur tous les exemples du dataset donné en
        prenant les exemples aléatoirement.
        :param desc_set: ndarray avec des descriptions
        :param label_set: ndarray avec les labels correspondants
        """
        for i in range(len(label_set)):
            x = desc_set[i]
            y = label_set[i]
            if self.score(x) * y <= 0:
                self.w += self.learning_rate * y * x

    def train(self, desc_set, label_set, niter_max=100, seuil=1e-5) -> list:
        """
        Apprentissage itératif du perceptron sur le dataset donné.
        :param desc_set: ndarray avec des descriptions
        :param label_set: ndarray avec les labels correspondants
        :param niter_max (par défaut: 100) : nombre d'itérations maximale
        :param seuil (par défaut: 0.01) : seuil de convergence
        :return liste des valeurs de norme de différences
        """
        dW = []
        last_w = self.w.copy()
        rdm = np.array(range(len(label_set)))
        for _ in range(niter_max):
            np.random.shuffle(rdm)
            self.train_step(desc_set[rdm], label_set[rdm])
            dW.append(np.linalg.norm(self.w - last_w))
            if dW[-1] < seuil: break
            last_w = self.w.copy()
        return dW

    def score(self, x):
        return np.dot(self.w, x)

    def predict(self, x):
        return -1 if self.score(x) <= 0 else 1


class Kernel():
    """
    Classe de base pour des fonctions noyau
    """

    def __init__(self, dim_in: int, dim_out: int):
        """
        :param dim_in : dimension de l'espace de départ (entrée du noyau)
        :param dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out

    def get_input_dim(self) -> int:
        """
        :return la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self) -> int:
        """
        :return la dimension de l'espace d'arrivée
        """
        return self.output_dim

    def transform(self, V: np.ndarray) -> np.ndarray:
        """
        :param V: données dans l'espace d'origine
        :return données transformées par le kernel
        """
        raise NotImplementedError("Please Implement this method")


class KernelBias(Kernel):
    """
    Classe pour un noyau simple 2D -> 3D
    Rajoute une colonne à 1
    """

    def transform(self, V):
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj


class KernelPoly(Kernel):
    """
    Kernel polynomial
    """

    def get_output_dim(self):
        d = self.input_dim
        return int(1 + 2*d + (d *(d-1)/2))

    def transform(self, X):
        n, d = X.shape
        d_ = int(1 + 2*d + (d *(d-1)/2))
        X_ = np.zeros((n, d_))
        X_[:, 0] = 1
        k = 2*d
        for j in range(d):
            X_[:, j+1] = X[:, j]
            X_[:, j+1+d] = X[:, j]**2
            for i in range(j+1,d):
                k += 1
                X_[:, k] = X[:, j] * X[:, i]
        return X_


class ClassifierPerceptronKernel(Classifier):
    """
    Perceptron de Rosenblatt kernelisé
    """

    def __init__(self, input_dimension, learning_rate, kernel, init=0):
        """
        :param input_dimension (int) : dimension de la description des exemples (espace originel)
        :param learning_rate : epsilon
        :param kernel : Kernel à utiliser
        :param init est le mode d'initialisation de w:
            - si 0 (par défaut): initialisation à 0 de w,
            - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        self.kernel = kernel
        kernel_dim = kernel.get_output_dim()
        if init == 0:
            self.w = np.zeros(kernel_dim)
        else:
            self.w = .001 * (2 * np.random.uniform(0, 1, kernel_dim) - 1)

    def train_step(self, desc_set, label_set):
        """
        Réalise une unique itération sur tous les exemples du dataset donné en
        prenant les exemples aléatoirement.
        :param desc_set: ndarray avec des descriptions dans l'espace originel
        :param label_set: ndarray avec les labels correspondants dans l'espace originel
        """
        for i in range(len(label_set)):
            x = desc_set[i]
            y = label_set[i]
            if np.dot(self.w, x) * y <= 0:
                self.w += self.learning_rate * y * x

    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """
        Apprentissage itératif du perceptron sur le dataset donné.
        :param desc_set: ndarray avec des descriptions dans l'espace originel
        :param label_set: ndarray avec les labels correspondants dans l'espace originel
        :param niter_max (par défaut: 100) : nombre d'itérations maximale
        :param seuil (par défaut: 0.01) : seuil de convergence
        :return liste des valeurs de norme de différences
        """
        desc_kernel_set = self.kernel.transform(desc_set)
        dW = []
        last_w = self.w.copy()
        rdm = np.array(range(len(label_set)))
        for _ in range(niter_max):
            np.random.shuffle(rdm)
            self.train_step(desc_kernel_set[rdm], label_set[rdm])
            dW.append(np.linalg.norm(self.w - last_w))
            if dW[-1] < seuil: break
            last_w = self.w.copy()
        return dW

    def score(self,x):
        """
        :param x: une description (dans l'espace originel)
        :return le score de prédiction sur x
        """
        return np.dot(self.w, self.kernel.transform(x.reshape(1,-1))[0])

    def predict(self, x):
        """
        :param x: une description (dans l'espace originel)
        :return la prediction sur x (soit -1 ou soit +1)
        """
        return -1 if self.score(x) <= 0 else +1


class ClassifierPerceptronBiais(Classifier):
    """
    Classifieur Perceptron à f(xi)*yi < 1
    """

    def __init__(self, input_dimension, learning_rate, init=0):
        """
        :param input_dimension (int) : dimension de la description des exemples (>0)
        :param learning_rate : epsilon
        :param init est le mode d'initialisation de w:
            - si 0 (par défaut): initialisation à 0 de w,
            - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        if init == 0:
            self.w = np.zeros(input_dimension)
        else:
            self.w = .001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.allw = []
        self.allw.append(self.w.copy())

    def get_allw(self):
        """
        :return la liste des différentes valeurs de w
        """
        return self.allw

    def train_step(self, desc_set: np.ndarray, label_set: np.ndarray):
        """
        Réalise une unique itération sur tous les exemples du dataset donné en
        prenant les exemples aléatoirement.
        :param desc_set: ndarray avec des descriptions
        :param label_set: ndarray avec les labels correspondants
        """
        for i in range(len(label_set)):
            x = desc_set[i]
            y = label_set[i]
            fx = self.score(x)
            if fx * y <= 1:
                self.w += self.learning_rate * (y - fx)  * x
                self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, niter_max=100, seuil=1e-5) -> list:
        """
        Apprentissage itératif du perceptron sur le dataset donné.
        :param desc_set: ndarray avec des descriptions
        :param label_set: ndarray avec les labels correspondants
        :param niter_max (par défaut: 100) : nombre d'itérations maximale
        :param seuil (par défaut: 0.01) : seuil de convergence
        :return liste des valeurs de norme de différences
        """
        dW = []
        last_w = self.w.copy()
        rdm = np.array(range(len(label_set)))
        for _ in range(niter_max):
            np.random.shuffle(rdm)
            self.train_step(desc_set[rdm], label_set[rdm])
            dW.append(np.linalg.norm(self.w - last_w))
            if dW[-1] < seuil: break
            last_w = self.w.copy()
        return dW

    def score(self, x):
        return np.dot(self.w, x)

    def predict(self, x):
        return -1 if self.score(x) <= 0 else 1
