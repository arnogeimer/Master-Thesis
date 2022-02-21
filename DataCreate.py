# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:37:54 2020

@author: ArNo1
"""

import numpy as np
import random
import matplotlib as plt
#### MODEL PARAMETERS

noise_variance = 1
mu = 0.
learningRate = 0.01
lamb = 0.1
thetavalue = 10
#random.seed(123)
#np.random.seed(123)

def CreateData(n, p1, sparsity, noise):
    
    X = np.random.normal(0, noise_variance, size = (n, p1))
    
    X_normalized =  X.T
    for i in range(len(X_normalized)):
        linenorm = np.linalg.norm(X_normalized[i], 2)
        X_normalized[i] /= linenorm
        
    theta = np.zeros([p1, 1])
    p2 = 2*p1
    
    sparsity_indexes = random.sample(range(1, p1), sparsity)
    
    for entry in sparsity_indexes:
        theta[entry] = thetavalue #np.random.uniform(0, 1)
    if noise:
        Y = X.dot(theta) + np.random.normal(0, noise_variance, size=(n, 1))
    else:
        Y = X.dot(theta)
    return X, Y, theta, sparsity_indexes

X, Y, theta, spars = CreateData(50, 100, 20, True)
