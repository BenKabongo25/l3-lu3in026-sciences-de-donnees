# -*- coding: utf-8 -*-

# Ben Kabongo B.
# Février 2022

# Sorbonne Université
# LU3IN026 - Sciences des données
# Classifieurs


import copy
import itertools
import numpy as np
import sys


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
    '''
    Kernel polynomial
    '''

    def __init__(self, in_dim):
        Kernel.__init__(self, in_dim, int(1 + 2*in_dim + (in_dim *(in_dim-1)/2)))

    def transform(self, X):
        n, d = X.shape
        d_ = int(1 + 2*d + (d *(d-1)/2))
        X_ = np.zeros((n, d_))
        X_[:, 0] = 1
        rng = np.arange(d)
        rng_ = np.arange(1 + 2*d, d_)
        comb = np.array(list(itertools.combinations(rng, 2)))
        X_[:, rng+1] = X[:, rng]
        X_[:, rng+1+d] = X[:, rng] ** 2
        X_[:, rng_] = X[:, comb[:,0]] * X[:, comb[:,1]]
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


class ClassifierMultiOAA(Classifier):
    """
    Classifier multi-classe générique
    """
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.classifiers = []
            
    def train(self, desc_set, label_set, **args):
        self.classifiers = [copy.deepcopy(self.classifier) for i in np.unique(label_set)]
        for i in range(len(self.classifiers)):
            label_set_i = label_set.copy()
            label_set_i[label_set_i != i] = -1
            label_set_i[label_set_i == i] = +1
            self.classifiers[i].train(desc_set, label_set_i, **args)
            
    def score(self, x):
        return np.array([cl.score(x) for cl in self.classifiers])

    def predict(self, x):
        return np.argmax(self.score(x))


class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        self.w = .001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.history = history
        self.allw = []
        if self.history:
            self.allw.append(self.w.copy())
        self.niter_max = niter_max
        
    def train(self, desc_set, label_set):
        #perfs = [0]
        for _ in range(self.niter_max):
            #rdm = np.array(range(len(label_set)))
            #np.random.shuffle(rdm)
            #for i in rdm:
            i = np.random.randint(0, len(label_set))
            xi = desc_set[i]
            yi = label_set[i]
            delta = xi.T * ((np.dot(xi, self.w) - yi) ** 2)
            self.w -= self.learning_rate * delta
            if self.history:
                self.allw.append(self.w.copy())
            # test de convergeance
            #perfs.append(self.accuracy(desc_set, label_set))
            #if np.abs(perfs[-1] - perfs[-2]) < 1e-5:
            #    break
        #return perfs
    
    def score(self,x):
        return np.dot(self.w, x)
    
    def predict(self, x):
        return -1 if self.score(x) <= 0 else +1


class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE Analytique
    """

    def __init__(self, input_dimension):
        Classifier.__init__(self, input_dimension)
        self.w = np.zeros(input_dimension)
        
    def train(self, desc_set, label_set):
        X = desc_set
        Y = label_set
        self.w = np.linalg.solve(X.T @ X, X.T @ Y)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x) <= 0 else +1


def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    u, c = np.unique(Y, return_counts=True)
    return u[np.argmax(c)]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    np.seterr(divide='ignore', invalid='ignore')
    return np.sum(P * -np.nan_to_num(np.log(P)))


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    u, c = np.unique(Y, return_counts=True)
    P = c / len(Y)
    return shannon(P)


class Noeud:
    """ Classe de base des noeuds des arbres de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att
        if (nom == ''): 
            nom = 'att_'+str(num_att)
        self.nom_attribut = nom 
        self.Les_fils = None
        self.classe   = None
    
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (Noeud) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = fils

    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        raise NotImplementedError

    def to_graph(self, g, prefixe='A'):
        raise NotImplementedError


class NoeudCategoriel(Noeud):
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
        
    def classifie(self, exemple):
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.min  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None
        
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        for i in range(nb_col):
            gain = 0
            U, C = np.unique(X[:, i], return_counts=True)
            P = C / np.sum(C)
            for j in range(len(U)):
                uuY, cuY = np.unique(Y[np.where(X[:,i] == U[j])[0]], return_counts=True)
                puY = cuY / np.sum(cuY)
                gain += P[j] * shannon(puY)
            gain = entropie_classe - gain
            if gain > gain_max:
                gain_max = gain
                i_best = i
                Xbest_valeurs = U
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


class NoeudNumerique(Noeud):
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        Noeud.__init__(self, num_att, nom)
        self.seuil = None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
            
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)
        return self.Les_fils['sup'].classifie(exemple)
          
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)


def partitionne(m_desc, m_class, n, s):
    w = np.where(m_desc[:,n] <= s)[0]
    left_desc, left_class = m_desc[w], m_class[w]
    w = np.where(m_desc[:,n] > s)[0]
    right_desc, right_class = m_desc[w], m_class[w]
    return ((left_desc, left_class), (right_desc, right_class))


def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_set = None
        
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        
        for i in range(nb_col):
            gain = 0
            ((seuil, entropie), (_, _)) = discretise(X, Y, i)
            partition = ((X, Y), (None, None))
            if seuil is not None:
                partition = partitionne(X, Y, i, seuil)
            gain = entropie_classe - entropie
            if gain > gain_max:
                gain_max = gain
                i_best = i
                Xbest_tuple = partition
                Xbest_seuil = seuil
        
        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil,
                              construit_AD_num(left_data,left_class, epsilon, LNoms),
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud


class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
