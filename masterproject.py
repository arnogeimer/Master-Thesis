# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:12:39 2020

@author: ArNo1
"""

import sys
import numpy as np
import random
import matplotlib as plt #pip install matplotlib
import statsmodels.api as sm #pip install statsmodels
import scipy.stats as ss
import sklearn.preprocessing #pip install sklearn
from sklearn import linear_model
import pickle
#import Main
import cvxopt

import time


### Functions

path = 'C:/Users/ArNo1/Documents/Uni/Master/'

threshhold = 1e-6
def threshold(array, thresh):
    return [0 if abs(a)<thresh else a for a in array]

def TPR(theta, thetahat):  #True Positive Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    PR = 0
    positives = 0
    for i in range(len(theta)):
        if theta[i] != 0 :
             positives += 1
             if thetahat[i] > 0:
                 PR += 1
    if positives == 0:
        return 1
    return PR/positives

def FPR(theta, thetahat):  #False Positive Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    NR = 0
    positives = 0
    for i in range(len(theta)):
        if thetahat[i] > 0 :
             positives += 1
             if theta[i] == 0:
                 NR += 1
    if positives == 0:
        return 0
    return NR/positives

def TNR(theta, thetahat):  #True Negative Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    NR = 0
    negatives = 0
    for i in range(len(theta)):
        if theta[i] == 0 :
             negatives += 1
             if thetahat[i] == 0:
                 NR += 1
    if negatives == 0:
        return 1
    return NR/negatives

def FNR(theta, thetahat):  #False Negative Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    NR = 0
    negatives = 0
    for i in range(len(theta)):
        if theta[i] != 0 :
             negatives += 1
             if thetahat[i] == 0:
                 NR += 1
    if negatives == 0:
        return NR
    return NR/negatives

def ESR(theta, thetahat):  #exact support recovery
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    esr = True
    i = 0
    while esr and i < len(thetahat):
        if theta[i] == 0 and thetahat[i] != 0:
            esr = False
        elif theta[i] != 0 and thetahat[i] == 0:
            esr = False
        i += 1
    return esr

def sparse_multi(X, v):
    n, p1 = X.shape
    if p1 != len(v):
        raise ValueError('Dimension mismatch')
        
    indices = np.nonzero(v)
    X_ = np.matrix([X[:,ind] for ind in indices][0])
    v_ = np.array([v[ind] for ind in indices][0])
    return X_@v_

### Important Stuff

#np.random.seed(123)
#random.seed(123)

thetavalue = 10
noise_variance = 1
alpha = 0.05
noise = True


class Model:
    
    def __init__(self, X, s, method): # methods = ['sqrtLASSO', 'LASSO', 'ANN']
        
        self.X = X
        self.s = s
        self.method = method
        self.n, self.p1 = X.shape
        
        self.normalize
        
        self.Ys = []
        self.thetas = []
        self.sparseindexes = []
        
        self.thetahats = []
        self.thetahats_pred = []
        self.simulcount = 0
        self.lamdaqut = None
        
        self.learnt = False
        
    @property
    
    def normalize(self):
        self.X = sklearn.preprocessing.normalize(self.X, norm='l2', axis=0, copy=True, return_norm=False)
    
    def simulate(self, count): #Simulate /count instances of theta and corresponding Y for X, s
        for i in range(count):
            sparsity_indexes = random.sample(range(1, self.p1), self.s)
            theta = np.array([thetavalue*np.int(i in sparsity_indexes) for i in range(0, self.p1)])
            Y = self.X.dot(theta) + noise*np.random.normal(0, noise_variance, size=(self.n))
            self.Ys.append(Y)
            self.thetas.append(theta)
            self.sparseindexes.append(sparsity_indexes)
            self.simulcount += 1
            
    def approxlam0(self): #Simulate /sims H0's and calculate corresponding lamda0, then lamdaqut = (1-alpha)quantile
        
        lamda0s = []
        sims = 500
        
        if self.method == 'sqrtLASSO':
            for i in range(sims):
                Y = np.random.normal(0, noise_variance, size=(self.n))
                Ym = Y - 1/np.sqrt(self.n)*np.mean(Y)
                lam0 = np.sqrt(self.n)*np.linalg.norm(self.X.T.dot(Ym), np.inf)/np.linalg.norm(Ym, 2)
                lamda0s.append(lam0)
                
        if self.method == 'LASSO':
             for i in range(sims):
                Y = np.random.normal(0, noise_variance, size=(self.n))
                Ym = Y - np.mean(Y)
                lam0 = (1/self.n)*np.linalg.norm(self.X.T.dot(Ym), np.inf)
                lamda0s.append(lam0)
                
        if self.method == 'ANN':
            for i in range(sims):
                Y = np.random.normal(0, noise_variance, size=(self.n))
                lam0 = np.linalg.norm(self.X.T.dot(Y), np.inf)/np.linalg.norm(Y, 2)
                lamda0s.append(lam0)
                
        self.lamdaqut = np.quantile(lamda0s, 1-alpha)
        #print('\u03BB =', self.lamdaqut)
            
    def learn(self): #Loops over the simuated data and calculates method prediction (saved @thetahats)
             
        if self.method == 'sqrtLASSO':
            if self.lamdaqut == None:
                self.approxlam0()
            for y in self.Ys:
                thetahat = sm.regression.linear_model.OLS(y, self.X).fit_regularized(method  = 'sqrt_lasso', alpha = self.lamdaqut).params
                self.thetahats.append(thetahat)
            self.learnt = True
            
        if self.method == 'LASSO':
            if self.lamdaqut == None:
                self.approxlam0()
            for y in self.Ys:
                clf = linear_model.Lasso(alpha=self.lamdaqut, fit_intercept = True, positive = False)
                clf.fit(self.X, y)
                thetahat = clf.coef_
                self.thetahats.append(thetahat)
            self.learnt = True
            
        if self.method == 'ANN':
            if self.lamdaqut == None:
                self.approxlam0()
            i = 0
            for y in self.Ys:
                thetahat = Main.TFmodel(self.X,y,Main.softplus,self.lamdaqut,0.25,thetavalue,self.sparseindexes[i],True,ifTrain=True)
                self.thetahats.append(thetahat)
                i += 1
            self.learnt = True
            
    def bestlamdapred(self):
        
        if not self.learnt:
            self.approxlam0()
            if len(self.Ys) == 0:
                self.simulate(1)
        
        lampreds = []
        
        lamrange = np.linspace(self.lamdaqut, 0, 100)[:-1]
        Xtest = np.random.normal(size = (self.n**2, self.p1)) 
        self.best_lamda_preds = []
        self.pred_errors = [[] for i in range(self.simulcount)]
        
        if self.method == 'LASSO':
            
            maxiter = 1250
            if self.s == 0:
                maxiter = 1500

            for i in range(self.simulcount): #loop over thetas/ys
                pred_err_max = np.infty
                y_pred = Xtest.dot(self.thetas[i])
                for lamda in lamrange:
                    clf = linear_model.Lasso(alpha=lamda, fit_intercept = True, positive = False, warm_start = True, max_iter = maxiter)
                    clf.fit(self.X, self.Ys[i])
                    thetahat = clf.coef_
                    error = np.linalg.norm(clf.intercept_ + sparse_multi(Xtest,thetahat) - y_pred)
                    if error < pred_err_max:
                        pred_err_max = error
                        best_lamda_pred = lamda
                    self.pred_errors[i].append(error)
                self.best_lamda_preds.append(best_lamda_pred)
                
        
    def learn_predictive(self):
        
        self.bestlamdapred()
        
        self.intercepts_pred = []
        for i in range(self.simulcount):
            if self.method == 'LASSO':
                clf = linear_model.Lasso(alpha=self.best_lamda_preds[i], fit_intercept = True, positive = False)
                clf.fit(self.X, self.Ys[i])
                self.intercepts_pred.append(clf.intercept_)
                thetahat = clf.coef_
                self.thetahats_pred.append(thetahat)
                 
    def save(self):
        if self.learnt:
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(self.method, self.n, self.p1, self.s), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)  
        else:
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(self.n, self.p1, self.s), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)  

x = Model(np.random.normal(size = (100, 150)), s = 20, method = 'LASSO')
x.simulate(100)
x.learn_predictive()

def LASSOPATH(x, j):
    lamrange = np.linspace(x.lamdaqut, 0, 100)[:-1]
    thetahats = []
    for i in range(len(lamrange)):
        clf = linear_model.Lasso(alpha=lamrange[i], fit_intercept = True, warm_start = True)
        clf.fit(x.X, x.Ys[j])
        thetahat = clf.coef_
        thetahats.append(thetahat)
    for entry in np.matrix(thetahats).T:
        plt.pyplot.plot(lamrange, entry.T)
        
n = 100
p1range = np.flip((1000/np.linspace(1, 10, 20)).astype(int))
srange = np.linspace(0, 99, 50).astype(int)
methods = ['sqrtLASSO', 'LASSO', 'ANN']


'''
def createData(n, p1range, srange): 
    xvals = [np.random.normal(0, 1, size = (n, p1)) for p1 in p1range]
    for x in xvals:
        for s in srange:
            ct = Model(x, s, None)
            ct.simulate(50)
            ct.save()
        
def SimulateThread(s, method):
    for i in range(len(p1range)):
        with open(path+'Data/n = {}/p = {} s = {}.pkl'.format(n, p1range[i], s), 'rb') as f:
            ct = pickle.load(f)
            ct.method = method
        ct.learn()
        ct.save()
        print('Learnt p1 = {} with s = {} \n'.format(p1range[i], s))

for s in srange:
    SimulateThread(s, method)
    print('>>>Calculated' , str(s), '\n')


# Multithreading part
import threading

#createData(n, p1range, srange)
if __name__ == "__main__":
    threads = list()
    for s in srange:
        thrd = threading.Thread(target=SimulateThread, args=(s, method,))
        threads.append(thrd)
        thrd.start()
    print('Finished')
'''

def getRates(function,method,n,p1range,srange):
    sr = len(srange)
    pr = len(p1range)
    rates = np.zeros((sr, pr))
    for i in range(sr):
        for j in range(pr):
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(method,n, p1range[j], srange[i]), 'rb') as input:
                rate = 0
                ctnew = pickle.load(input)
                for k in range(ctnew.simulcount):
                    rate += function(ctnew.thetas[k], ctnew.thetahats[k])/ctnew.simulcount
                rates[i][j] = rate
    print(rates)
    return rates

def getpredictiveRates(function,method,n,p1range,srange):
    sr = len(srange)
    pr = len(p1range)
    rates = np.zeros((sr, pr))
    for i in range(sr):
        for j in range(pr):
            print('S = {} p = {}'.format(2*i, j))
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(method,n, p1range[j], srange[i]), 'rb') as input:
                rate = 0
                ctnew = pickle.load(input)
                #ctnew.learn()
                #ctnew.pred_errors = []
                #ctnew.best_lamda_pred = None
                #ctnew.thetahats_pred = []
                #ctnew.learn_predictive()
                
                ctnew.save()
                for k in range(ctnew.simulcount):
                    rate += function(ctnew.thetas[k], ctnew.thetahats_pred[k])/ctnew.simulcount
                rates[i][j] = rate
    print(rates)
    return rates
    
def pickleopen(n, p, s, method):
    with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(method,n, p, s), 'rb') as f:
        x = pickle.load(f)
    return x


method = 'LASSO'
function = FPR
#trus = getpredictiveRates(function,method,n,p1range,srange)
trus[-2][-2] = 1
mini, maxi = round(np.min(trus), 3), round(np.max(trus), 3)
plt.pyplot.pcolormesh(n/np.array(p1range), srange/n, trus)
#plt.pyplot.pcolormesh(trus)
plt.pyplot.title('{} with min {} and max {} for {}'.format(function.__code__.co_name, mini, maxi, method))
plt.pyplot.xlabel('n/p1')
plt.pyplot.ylabel('s/n')


'''
np.random.seed(123)
n = 1000
p1 = 250
X = np.random.normal(0, 1, size = (n, p1))

theta = np.zeros(p1)
for i in range(25, 55):
    theta[i]  = 10

y_ = X.dot(theta)
y = X.dot(theta) + np.random.normal(0, 1, n)
#plt.pyplot.plot(theta, color = 'r')

Ym = y - 1/np.sqrt(n)*np.mean(y)
lam0 = np.sqrt(n)*np.linalg.norm(X.T.dot(Ym), np.inf)/np.linalg.norm(Ym, 2)
#print(lam0)



#def findlamopt(X, y, lamdas, method):
    
def findlamopt(X, y, method):
    
    def goldensect(func, xrange):
    
        if method == 'LASSO':
            eps = .000001
        if method == 'sqrtLASSO':
            eps = .1
            
        a = xrange[0]
        b = xrange[1]
        tau = (3-np.sqrt(5))/2
        while np.abs(b-a) > eps:
            print(np.abs(b-a))
            x = a + tau*(b-a)
            y = b - tau*(b-a)
            gx = func(x)
            gy = func(y)
            if gx < gy:
                if func(a)<=gx:
                    a = a
                    b = x
                if func(a)>gx:
                    a = a
                    b = y
            if gx > gy:
                if gy>=func(b):
                    a = y
                    b = b
                if gy<func(b):
                    a = x
                    b = b
            if gx == gy:
                a = x
                b = y
        return (a+b)/2 
    
    if method == 'sqrtLASSO':
        
        def sqrtL(lam):
            
            thetahat = sm.regression.linear_model.OLS(y, X).fit_regularized(method  = 'sqrt_lasso', alpha = lam).params
            return np.linalg.norm(X.dot(thetahat) - y_, ord = 2)
        
        #lamopt = goldensect(sqrtL, (3**(np.log10(n)))*np.array([10, 20]))
        lamopt = goldensect(sqrtL, [0, n])
        print(lamopt)
    
    if method == 'LASSO':
        def L(lam):
            
            clf = linear_model.Lasso(alpha=lam, fit_intercept = True, positive = True)
            clf.fit(X, y)
            thetahat = clf.coef_
            return np.linalg.norm(X.dot(thetahat) - y_, ord = 2)
        
        #lamopt = goldensect(L, (3**(np.log10(n)))*np.array([10, 20]))
        lamopt = goldensect(L, [0.001, 1])
        print(lamopt)
    


    errs = []
    errmin = np.infty
    lamopt = None
    
    if method == 'sqrtLASSO':
        
        for lam in lamdas:
            
            thetahat = sm.regression.linear_model.OLS(y, X).fit_regularized(method  = 'sqrt_lasso', alpha = lam).params
            #plt.pyplot.plot(thetahat)
            
            error = np.linalg.norm(X.dot(thetahat) - y_, ord = 2)
            print(error)
            if error < errmin:
                errmin = error
                lamopt = lam
            errs.append(error)
        plt.pyplot.plot(lamdas, errs, color = 'pink')
            
    if method == 'LASSO':

        for lam in lamdas:
            
            clf = linear_model.Lasso(alpha=lam, fit_intercept = True, positive = True)
            clf.fit(X, y)
            thetahat = clf.coef_
            #plt.pyplot.plot(thetahat)
            
            error = np.linalg.norm(X.dot(thetahat) - y_, ord = 2)
            print(error)
            if error < errmin:
                errmin = error
                lamopt = lam
            errs.append(error)
        plt.pyplot.plot(lamdas, errs, color = 'orange')
        
    print('The optimal lambda is {} with an error of {}'.format(lamopt, errmin))
'''

'''
#findlamopt(X, y, (1/3)**(np.log10(n)-2)*np.linspace(.05,.2,200), 'LASSO')
#findlamopt(X, y, 'LASSO') 
#findlamopt(X, y, np.linspace(30,50,25), 'sqrtLASSO')
#findlamopt(X, y, 'sqrtLASSO')

def LASSOpath(X, y):
    
    lamqut = (1/n)*np.linalg.norm(X.T.dot(y - np.mean(y)), ord = np.infty)
    #lamdarange = np.linspace(lamqut, 0, 1001)[:-1]
    lamdarange = np.logspace(np.log(lamqut), 0, 1001, base = np.e)[:-1] - 1
    alphahats = []
    for lam in lamdarange:
        clf = linear_model.Lasso(alpha=lam, fit_intercept = True, positive = True, warm_start = True)
        clf.fit(X, y)
        alphahat = clf.coef_
        alphahats.append(alphahat)
    for entry in np.array(alphahats).T:
        #plt.pyplot.plot(lamdarange, entry, color = 'black')
        pass
    return alphahats
'''
'''
alphahats = LASSOpath(X,y)
Xb = np.random.normal(0, 1, size = (50*n, p1))
errs = []
for entry in alphahats:
    error = np.linalg.norm(Xb.dot(entry) - Xb.dot(theta), ord = 2)
    errs.append(error)
plt.pyplot.plot(errs)
'''

    
