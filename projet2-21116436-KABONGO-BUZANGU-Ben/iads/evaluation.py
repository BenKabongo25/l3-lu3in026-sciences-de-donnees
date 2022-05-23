# -*- coding: utf-8 -*-

# Ben Kabongo B.
# Février 2022

# Sorbonne Université
# LU3IN026 - Sciences des données
# Evaluation des Classifieurs

import numpy as np


def analyse_perfs(perf):
    return np.mean(perf), np.var(perf)